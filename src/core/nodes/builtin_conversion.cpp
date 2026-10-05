/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"
#include "core/tensor_fused.hpp"
#include "core/tensor_spatial.hpp"

#include <algorithm>
#include <bit>
#include <numbers>
#include <vector>

namespace lfs::nodes::builtin {
    namespace {
        // Surface samples drawn from a counter hash of (seed, sample, stream) rather than a backend's
        // generator, so a seed places the same samples on every backend. The top 24 hash bits make an
        // exact float in [0, 1) everywhere.
        template <typename Word>
        Word hash32(Word x) {
            x = (x ^ (x >> 17u)) * 0xed5ad4bbu;
            x = (x ^ (x >> 11u)) * 0xac4c1b51u;
            x = (x ^ (x >> 15u)) * 0x31848babu;
            return x ^ (x >> 14u);
        }

        template <typename Word>
        auto unit_draw(const Word& sample, const Word& key, const uint32_t stream) {
            const Word bits = hash32<Word>(hash32<Word>(sample * 3u + stream) ^ key) >> 8u;
            if constexpr (std::is_same_v<Word, uint32_t>)
                return static_cast<float>(bits) * 0x1p-24f;
            else
                return bits.cast(DataType::Float32) * 0x1p-24f;
        }

        uint32_t sampling_key(const uint64_t seed) {
            return hash32(static_cast<uint32_t>(seed) ^ hash32(static_cast<uint32_t>(seed >> 32U) + 0x9e3779b9u));
        }

        // Face by inverse CDF over the cumulative areas, as a binary search unrolled over the bits of the
        // face count, plus the two barycentric draws.
        core::fused::Kernel make_surface_sampler(const int steps) {
            namespace f = core::fused;
            f::Builder builder(1);
            const auto cdf = builder.input(DataType::Float32, 1);
            const auto parameters = builder.input(DataType::Int32, 1);
            const auto key = parameters.at({0}).cast(DataType::UInt32);
            const auto faces = parameters.at({1});
            const auto sample = builder.iota(0).cast(DataType::UInt32);
            const auto target = unit_draw(sample, key, 0) * cdf.gather({faces - 1});
            auto below = builder.constant(0);
            for (int bit = steps - 1; bit >= 0; --bit) {
                const auto candidate = below + (1 << bit);
                below = f::where(candidate <= faces && cdf.gather({candidate - 1}, f::Bounds::Clamp) <= target,
                                 candidate, below);
            }
            builder.output(f::min(below, faces - 1), DataType::Int32);
            builder.output(f::sqrt(unit_draw(sample, key, 1)), DataType::Float32);
            builder.output(unit_draw(sample, key, 2), DataType::Float32);
            return f::Kernel(builder);
        }

        struct SurfaceSamples {
            Tensor faces; // Int32
            Tensor root;  // square root of the first barycentric draw
            Tensor second;
        };

        SurfaceSamples sample_surface(const Tensor& areas, const size_t count, const uint64_t seed) {
            const size_t face_count = areas.numel();
            const auto key = sampling_key(seed);
            const auto cdf = areas.cumsum(0);
            if (areas.device() == Device::GPU) {
                static const auto kernels = [] {
                    std::vector<core::fused::Kernel> result;
                    for (int steps = 1; steps <= 31; ++steps)
                        result.push_back(make_surface_sampler(steps));
                    return result;
                }();
                const auto parameters = Tensor::from_vector(
                    std::vector<int>{std::bit_cast<int>(key), static_cast<int>(face_count)}, {2}, Device::CPU);
                auto outputs = kernels[std::bit_width(face_count) - 1]({count}, {cdf, parameters});
                return {std::move(outputs[0]), std::move(outputs[1]), std::move(outputs[2])};
            }
            const auto sums = cdf.to_vector();
            std::vector<int> faces(count);
            std::vector<float> root(count), second(count);
            for (size_t i = 0; i < count; ++i) {
                const auto sample = static_cast<uint32_t>(i);
                const float target = unit_draw(sample, key, 0) * sums.back();
                const auto face = std::upper_bound(sums.begin(), sums.end(), target) - sums.begin();
                faces[i] = static_cast<int>(std::min<size_t>(face, face_count - 1));
                root[i] = std::sqrt(unit_draw(sample, key, 1));
                second[i] = unit_draw(sample, key, 2);
            }
            return {Tensor::from_vector(faces, {count}, Device::CPU),
                    Tensor::from_vector(root, {count}, Device::CPU),
                    Tensor::from_vector(second, {count}, Device::CPU)};
        }

        Tensor auto_point_radius(const Tensor& positions) {
            const size_t count = positions.shape()[0];
            if (count < 2)
                return Tensor::full({count}, 1e-6f, positions.device());
            // Estimate a robust cell width from a bounded, evenly spaced sample.
            // Outliers must not make all of a dense reconstruction share one cell.
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
            auto radii = core::point_neighbor_spacing(positions, width) * 0.5f;
            const auto sorted = radii.sort().first;
            const auto median = sorted.slice(0, (count - 1) / 2, count / 2 + 1).mean().maximum(1e-6f);
            return radii.maximum(median * 0.25f).minimum(median * 4.0f).maximum(1e-6f);
        }

        Tensor cross(const Tensor& a, const Tensor& b) {
            return Tensor::stack({channel(a, 1) * channel(b, 2) - channel(a, 2) * channel(b, 1),
                                  channel(a, 2) * channel(b, 0) - channel(a, 0) * channel(b, 2),
                                  channel(a, 0) * channel(b, 1) - channel(a, 1) * channel(b, 0)},
                                 1);
        }

        Tensor normalized(const Tensor& value) {
            return value / (value * value).sum(1, true).sqrt().maximum(1e-12f);
        }

        Tensor surface_rotation(const Tensor& normal) {
            const auto device = normal.device();
            const auto helper = Tensor::where(channel(normal, 2).abs().lt(0.9f).unsqueeze(1),
                                              vector_tensor({0, 0, 1}, device), vector_tensor({0, 1, 0}, device));
            const auto tangent = normalized(cross(helper, normal));
            const auto bitangent = cross(normal, tangent);
            // Matrix columns are the local X, Y and Z axes. Choose the largest
            // quaternion component to avoid division by zero at half turns.
            const auto xx = channel(tangent, 0);
            const auto yy = channel(bitangent, 1);
            const auto zz = channel(normal, 2);
            const auto w = (xx + yy + zz + 1).maximum(0).sqrt() * 0.5f;
            const auto x = (xx - yy - zz + 1).maximum(0).sqrt() * 0.5f;
            const auto y = (yy - xx - zz + 1).maximum(0).sqrt() * 0.5f;
            const auto z = (zz - xx - yy + 1).maximum(0).sqrt() * 0.5f;
            const auto a = channel(bitangent, 2) - channel(normal, 1);
            const auto b = channel(normal, 0) - channel(tangent, 2);
            const auto c = channel(tangent, 1) - channel(bitangent, 0);
            const auto xy = channel(bitangent, 0) + channel(tangent, 1);
            const auto xz = channel(normal, 0) + channel(tangent, 2);
            const auto yz = channel(normal, 1) + channel(bitangent, 2);
            const auto qw = Tensor::stack({w * w * 4, a, b, c}, 1) / (w.maximum(1e-8f) * 4).unsqueeze(1);
            const auto qx = Tensor::stack({a, x * x * 4, xy, xz}, 1) / (x.maximum(1e-8f) * 4).unsqueeze(1);
            const auto qy = Tensor::stack({b, xy, y * y * 4, yz}, 1) / (y.maximum(1e-8f) * 4).unsqueeze(1);
            const auto qz = Tensor::stack({c, xz, yz, z * z * 4}, 1) / (z.maximum(1e-8f) * 4).unsqueeze(1);
            return normalized(Tensor::where(w.ge(x.maximum(y).maximum(z)).unsqueeze(1), qw,
                                            Tensor::where(x.ge(y.maximum(z)).unsqueeze(1), qx,
                                                          Tensor::where(y.ge(z).unsqueeze(1), qy, qz))));
        }

        Tensor mesh_colours(const MeshComponent& component, const Tensor& faces,
                            const Tensor& vertices, const Tensor& weights) {
            const auto& mesh = *component.mesh;
            const size_t count = faces.numel();
            const auto device = mesh.vertices.device();
            const auto interpolate = [&](const Tensor& attribute) {
                return (attribute.index_select(0, vertices.flatten()).reshape(core::TensorShape{count, 3, attribute.shape()[1]}) * weights.unsqueeze(2)).sum(1);
            };
            if (mesh.has_colors())
                return interpolate(mesh.colors.slice(1, 0, 3));
            auto colours = Tensor::full({count, 3}, 0.5f, device);
            Tensor uv;
            if (mesh.has_texcoords())
                uv = interpolate(mesh.texcoords);
            const auto material_colour = [&](size_t index) {
                const auto& material = mesh.materials[index];
                auto base = vector_tensor(glm::vec3(material.base_color), device);
                if (!uv.is_valid() || material.albedo_tex == 0 || material.albedo_tex > mesh.texture_images.size())
                    return base;
                const auto& image = mesh.texture_images[material.albedo_tex - 1];
                if (image.width <= 0 || image.height <= 0 || image.channels < 3 ||
                    image.pixels.size() != static_cast<size_t>(image.width) * image.height * image.channels)
                    return base;
                const auto& texture = component.textures.at(material.albedo_tex - 1);
                if (!texture.is_valid())
                    return base;
                // The loader applies aiProcess_FlipUVs; image row zero is top.
                // Repeat addressing matches mesh rendering, with no second V flip.
                const auto wrapped = uv - uv.floor();
                const auto x = (channel(wrapped, 0) * static_cast<float>(image.width)).floor().clamp(0, image.width - 1).to(DataType::Int32);
                const auto y = (channel(wrapped, 1) * static_cast<float>(image.height)).floor().clamp(0, image.height - 1).to(DataType::Int32);
                return texture.index_select(0, y * image.width + x) * base;
            };
            if (mesh.submeshes.empty() && !mesh.materials.empty())
                return material_colour(0).expand({static_cast<int>(count), 3}).contiguous();
            for (const auto& submesh : mesh.submeshes) {
                if (submesh.material_index >= mesh.materials.size())
                    continue;
                const auto mask = faces.ge(static_cast<int>(submesh.start_index / 3))
                                      .logical_and(faces.lt(static_cast<int>((submesh.start_index + submesh.index_count) / 3)));
                colours = Tensor::where(mask.unsqueeze(1), material_colour(submesh.material_index), colours);
            }
            return colours;
        }
    } // namespace

    void evaluate_points_to_splats(NodeContext& context) {
        auto geometry = geometry_input(context);
        if (geometry.points) {
            const auto& points = *geometry.points;
            const auto count = points.positions.shape()[0];
            const auto device = points.positions.device();
            const float radius = input_float(context, "Radius");
            const float opacity = std::clamp(input_float(context, "Opacity", 1), 1e-6f, 1 - 1e-6f);
            const auto radii = radius <= 0 ? auto_point_radius(points.positions) : Tensor::full({count}, std::max(radius, 1e-6f), device);
            const auto rotation =
                Tensor::cat({Tensor::ones({count, 1}, device), Tensor::zeros({count, 3}, device)}, 1);
            geometry.splats =
                SplatsComponent{points.positions,
                                (points.colors - 0.5f) / kShC0,
                                Tensor::zeros({count, 0, 3}, device),
                                radii.log().unsqueeze(1).expand({static_cast<int>(count), 3}).contiguous(),
                                rotation,
                                Tensor::full({count}, std::log(opacity / (1 - opacity)), device),
                                0,
                                1,
                                points.attributes};
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void evaluate_mesh_to_splats(NodeContext& context) {
        auto geometry = geometry_input(context);
        if (!geometry.mesh || !geometry.mesh->mesh) {
            context.set_output("Geometry", std::move(geometry));
            return;
        }
        const auto& mesh = *geometry.mesh->mesh;
        const auto device = mesh.vertices.device();
        const size_t face_count = mesh.face_count();
        if (!face_count) {
            geometry.mesh.reset();
            context.set_output("Geometry", std::move(geometry));
            return;
        }
        const auto corners = mesh.vertices.index_select(0, mesh.indices.flatten()).reshape(core::TensorShape{face_count, 3, 3});
        const auto a = corners.slice(1, 0, 1).squeeze(1);
        const auto b = corners.slice(1, 1, 2).squeeze(1);
        const auto c = corners.slice(1, 2, 3).squeeze(1);
        const auto normals = cross(b - a, c - a);
        const auto areas = (normals * normals).sum(1).sqrt() * 0.5f;
        const float total_area = areas.sum().item<float>();
        if (!std::isfinite(total_area))
            throw NodeError("Mesh surface area must be finite");
        const float density = input_float(context, "Density");
        if (!std::isfinite(density) || density < 0)
            throw NodeError("Mesh sampling density must be finite and non-negative");
        const double requested = std::ceil(static_cast<double>(total_area) * density);
        const size_t count = static_cast<size_t>(std::clamp(requested, 0.0, static_cast<double>(input_int(context, "Max Count", 2000000))));
        if (!count) {
            geometry.mesh.reset();
            geometry.splats.reset();
            context.set_output("Geometry", std::move(geometry));
            return;
        }
        const auto seed = static_cast<uint64_t>(property_int(context, "seed"));
        const auto [faces, root, second] = sample_surface(areas, count, seed);
        const auto weights = Tensor::stack({root.neg() + 1, root * (second.neg() + 1), root * second}, 1);
        const auto vertices = mesh.indices.index_select(0, faces);
        const auto positions = (mesh.vertices.index_select(0, vertices.flatten()).reshape(core::TensorShape{count, 3, 3}) * weights.unsqueeze(2)).sum(1);
        auto normal = normalized(normals.index_select(0, faces));
        if (mesh.has_normals()) {
            const auto interpolated = (mesh.normals.index_select(0, vertices.flatten()).reshape(core::TensorShape{count, 3, 3}) * weights.unsqueeze(2)).sum(1);
            normal = Tensor::where((interpolated * interpolated).sum(1, true).gt(1e-16f), normalized(interpolated), normal);
        }
        const float r = 0.75f * std::sqrt(total_area / static_cast<float>(count));
        const float opacity = std::clamp(input_float(context, "Opacity", 0.95f), 1e-6f, 1 - 1e-6f);
        geometry.splats = SplatsComponent{
            positions,
            (mesh_colours(*geometry.mesh, faces, vertices, weights) - 0.5f) / kShC0,
            Tensor::zeros({count, 0, 3}, device),
            vector_tensor({std::log(r), std::log(r), std::log(0.1f * r)}, device).expand({static_cast<int>(count), 3}).contiguous(),
            surface_rotation(normal),
            Tensor::full({count}, std::log(opacity / (1 - opacity)), device),
            0,
            1,
            {}};
        geometry.mesh.reset();
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
        register_type(registry, type("lfs.points_to_splats", "Conversion",
                                     geometry_inputs({in("Radius", f, 0.0f).minimum(0).step_size(0.01),
                                                      in("Opacity", f, 0.9f).range(0, 1).step_size(0.01)}),
                                     {out("Geometry", geo)}, evaluate_points_to_splats));
        register_type(registry, type("lfs.splats_to_points", "Conversion",
                                     geometry_inputs({}), {out("Geometry", geo)}, evaluate_splats_to_points));
        register_type(registry, type("lfs.mesh_to_points", "Conversion",
                                     geometry_inputs({}), {out("Geometry", geo)}, evaluate_mesh_to_points));
        register_type(registry, type("lfs.mesh_to_splats", "Conversion",
                                     geometry_inputs({in("Density", f, 20000.0f / (4.0f * std::numbers::pi_v<float>)).minimum(0).step_size(10),
                                                      in("Max Count", std::string(INT_SOCKET), std::int64_t(2000000)).range(0, 2000000).step_size(1000),
                                                      in("Opacity", f, 0.95f).range(0, 1).step_size(0.01)}),
                                     {out("Geometry", geo)}, evaluate_mesh_to_splats, {prop("seed", PropertyKind::Int, 0)}));
    }

} // namespace lfs::nodes::builtin
