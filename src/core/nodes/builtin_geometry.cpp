/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor_spatial.hpp"
#include <set>
namespace lfs::nodes::builtin {
    std::shared_ptr<core::MeshData> filter_mesh_faces(const core::MeshData& source, const Tensor& selection,
                                                      bool keep_selected_faces) {
        const auto indices = source.indices.to(DataType::Int32).contiguous();
        const auto selected_vertices = selection.to(DataType::Float32).contiguous();
        const auto selected_corners = selected_vertices.gather(0, indices.reshape({-1})).reshape(indices.shape());
        const auto touches_selection = selected_corners.max(1).gt(0);
        const auto keep_faces = keep_selected_faces ? touches_selection : touches_selection.logical_not();
        const auto kept_faces = keep_faces.nonzero().reshape({-1}).to(DataType::Int32);

        std::vector<core::Submesh> submeshes;
        std::size_t output_index = 0;
        for (const auto& source_submesh : source.submeshes) {
            const auto first_face = source_submesh.start_index / 3;
            const auto end_face = (source_submesh.start_index + source_submesh.index_count) / 3;
            const auto kept_index_count = static_cast<std::size_t>(
                                              keep_faces.slice(0, first_face, end_face)
                                                  .to(DataType::Int32)
                                                  .sum()
                                                  .item<int>()) *
                                          3;
            if (kept_index_count) {
                submeshes.push_back({output_index, kept_index_count, source_submesh.material_index});
                output_index += kept_index_count;
            }
        }
        if (source.submeshes.empty() && kept_faces.numel()) {
            submeshes.push_back({0, kept_faces.numel() * 3, 0});
        }
        auto mesh = std::make_shared<core::MeshData>();
        mesh->vertices = source.vertices;
        mesh->normals = source.normals;
        mesh->tangents = source.tangents;
        mesh->texcoords = source.texcoords;
        mesh->colors = source.colors;
        mesh->indices = indices.index_select(0, kept_faces).contiguous();
        mesh->materials = source.materials;
        if (mesh->materials.empty() && !submeshes.empty())
            mesh->materials.emplace_back();
        mesh->texture_images = source.texture_images;
        mesh->submeshes = std::move(submeshes);
        return mesh;
    }

    void evaluate_set_position(NodeContext& context) {
        Geometry geometry = geometry_input(context);
        if (geometry.splats && geometry.splats->means.shape()[0] != 0) {
            auto& s = *geometry.splats;
            auto fc = field_context(s);
            auto w = selection(context, "Selection", fc);
            auto p = context.evaluate_field("Position", fc, VECTOR_SOCKET);
            auto o = context.evaluate_field("Offset", fc, VECTOR_SOCKET);
            s.means = blend(s.means, p + o, w);
        }
        if (geometry.points) {
            auto& p = *geometry.points;
            auto fc = field_context(p);
            auto w = selection(context, "Selection", fc);
            auto value = context.evaluate_field("Position", fc, VECTOR_SOCKET);
            auto offset = context.evaluate_field("Offset", fc, VECTOR_SOCKET);
            p.positions = blend(p.positions, value + offset, w);
        }
        if (geometry.mesh && geometry.mesh->mesh) {
            auto fc = field_context(*geometry.mesh);
            auto w = selection(context, "Selection", fc);
            auto p = context.evaluate_field("Position", fc, VECTOR_SOCKET);
            auto o = context.evaluate_field("Offset", fc, VECTOR_SOCKET);
            auto m = copy_mesh(*geometry.mesh->mesh, blend(geometry.mesh->mesh->vertices, p + o, w),
                               geometry.mesh->mesh->indices);
            geometry.mesh = MeshComponent{std::move(m), geometry.mesh->textures, geometry.mesh->attributes};
        }
        context.set_output("Geometry", std::move(geometry));
    }

    static Tensor normalized(const Tensor& vector) {
        return safe_divide(vector, (vector * vector).sum(1, true).sqrt());
    }

    void evaluate_transform(NodeContext& context) {
        auto geometry = geometry_input(context);
        const auto translation = input_vector(context, "Translation");
        const auto rotation_value = input_vector(context, "Rotation");
        const float scale = input_float(context, "Scale", 1);
        if (translation == glm::vec3(0) && rotation_value == glm::vec3(0) && scale == 1.0f) {
            context.set_output("Geometry", std::move(geometry));
            return;
        }
        const auto rotation = rotation_matrix(rotation_value);
        const auto matrix =
            glm::translate(glm::mat4(1), translation) * rotation * glm::scale(glm::mat4(1), glm::vec3(scale));
        if (geometry.splats && geometry.splats->means.shape()[0] != 0) {
            auto& splats = *geometry.splats;
            (void)core::transform_canonical(splats.means, splats.rotation, splats.scaling, splats.sh0, splats.shN,
                                            splats.sh_degree, matrix);
            splats.scene_scale *= std::abs(scale);
        }
        const auto positions = [&](const Tensor& value) {
            return value.matmul(matrix_tensor(glm::mat3(matrix), value.device())) +
                   vector_tensor(translation, value.device());
        };
        if (geometry.points)
            geometry.points->positions = positions(geometry.points->positions);
        if (geometry.mesh && geometry.mesh->mesh)
            geometry.mesh->mesh = transform_mesh(*geometry.mesh->mesh, matrix);
        context.set_output("Geometry", std::move(geometry));
    }
    Geometry separate_geometry(const NodeContext& context, const Geometry& source, bool selected) {
        Geometry result;
        if (source.splats) {
            auto fc = field_context(*source.splats);
            auto mask = selection(context, "Selection", fc, true);
            if (!selected)
                mask = mask.logical_not();
            result.splats = filter_splats(*source.splats, mask);
        }
        if (source.points) {
            auto fc = field_context(*source.points);
            auto mask = selection(context, "Selection", fc, true);
            if (!selected)
                mask = mask.logical_not();
            result.points = filter_points(*source.points, mask);
        }
        if (source.mesh && source.mesh->mesh) {
            auto fc = field_context(*source.mesh);
            auto mask = selection(context, "Selection", fc, true);
            result.mesh = MeshComponent{filter_mesh_faces(*source.mesh->mesh, mask, selected), source.mesh->textures,
                                        source.mesh->attributes};
        }
        return result;
    }

    void evaluate_delete(NodeContext& context) {
        Geometry source = geometry_input(context);
        context.set_output("Geometry", separate_geometry(context, source, false));
    }
    void evaluate_separate_geometry(NodeContext& context) {
        Geometry source = geometry_input(context);
        context.set_output("Selection", separate_geometry(context, source, true));
        context.set_output("Inverted", separate_geometry(context, source, false));
    }

    template <typename Component>
    AttributeMap join_attributes(const std::vector<const Component*>& components, core::Device device) {
        std::set<std::string> names;
        for (auto* c : components)
            for (auto& [name, _] : c->attributes)
                names.insert(name);
        AttributeMap result;
        for (auto& name : names) {
            std::vector<core::Tensor> tensors;
            core::TensorShape shape;
            DataType dtype = DataType::Float32;
            for (auto* c : components)
                if (auto it = c->attributes.find(name); it != c->attributes.end()) {
                    shape = it->second.shape();
                    dtype = it->second.dtype();
                    break;
                }
            for (auto* c : components) {
                auto it = c->attributes.find(name);
                if (it != c->attributes.end())
                    tensors.push_back(it->second);
                else {
                    size_t n;
                    if constexpr (std::is_same_v<Component, SplatsComponent>)
                        n = c->means.shape()[0];
                    else if constexpr (std::is_same_v<Component, MeshComponent>)
                        n = static_cast<size_t>(c->mesh->vertex_count());
                    else
                        n = c->positions.shape()[0];
                    auto dimensions = shape.dims();
                    dimensions[0] = n;
                    tensors.push_back(core::Tensor::zeros(core::TensorShape(dimensions), device, dtype));
                }
            }
            result[name] = core::Tensor::cat(tensors, 0);
        }
        return result;
    }

    Geometry join_geometries(const std::vector<Geometry>& source_values) {
        auto values = source_values;
        GeometryDeviceCache devices;
        std::optional<core::GpuBackendScope> scope;
        std::optional<core::Device> device;
        for (const auto& value : values) {
            const Tensor* tensor = value.splats                     ? &value.splats->means
                                   : value.points                   ? &value.points->positions
                                   : value.mesh && value.mesh->mesh ? &value.mesh->mesh->vertices
                                                                    : nullptr;
            if (tensor) {
                device = tensor->device();
                if (const auto backend = core::gpu_backend_of(*tensor))
                    scope.emplace(*backend);
                break;
            }
        }
        if (device)
            for (auto& value : values)
                value = devices.convert(std::move(value), *device);
        Geometry result;
        std::vector<const SplatsComponent*> splats;
        std::vector<const PointsComponent*> points;
        std::vector<std::shared_ptr<const core::MeshData>> meshes;
        std::vector<const MeshComponent*> mesh_components;
        for (auto& g : values) {
            if (g.splats)
                splats.push_back(&*g.splats);
            if (g.points)
                points.push_back(&*g.points);
            if (g.mesh && g.mesh->mesh) {
                meshes.push_back(g.mesh->mesh);
                mesh_components.push_back(&*g.mesh);
            }
        }
        if (!splats.empty()) {
            int degree = 0;
            for (auto* s : splats)
                degree = std::max(degree, s->sh_degree);
            size_t coeff = (degree + 1) * (degree + 1) - 1;
            std::vector<core::Tensor> means, sh0, shn, scaling, rotation, opacity;
            for (auto* s : splats) {
                means.push_back(s->means);
                sh0.push_back(s->sh0);
                scaling.push_back(s->scaling);
                rotation.push_back(s->rotation);
                opacity.push_back(s->opacity);
                if (s->shN.shape()[1] == coeff)
                    shn.push_back(s->shN);
                else {
                    auto padded = core::Tensor::zeros({s->means.shape()[0], coeff, 3}, s->means.device());
                    if (s->shN.shape()[1])
                        padded.slice(1, 0, s->shN.shape()[1]).copy_from(s->shN);
                    shn.push_back(std::move(padded));
                }
            }
            result.splats = SplatsComponent{core::Tensor::cat(means, 0),
                                            core::Tensor::cat(sh0, 0),
                                            core::Tensor::cat(shn, 0),
                                            core::Tensor::cat(scaling, 0),
                                            core::Tensor::cat(rotation, 0),
                                            core::Tensor::cat(opacity, 0),
                                            degree,
                                            splats.front()->scene_scale,
                                            join_attributes(splats, splats.front()->means.device())};
        }
        if (!points.empty()) {
            std::vector<core::Tensor> pos, col;
            for (auto* p : points) {
                pos.push_back(p->positions);
                col.push_back(p->colors);
            }
            result.points = PointsComponent{core::Tensor::cat(pos, 0), core::Tensor::cat(col, 0),
                                            join_attributes(points, points.front()->positions.device())};
        }
        if (!meshes.empty()) {
            std::vector<core::Tensor> verts, idx;
            auto mesh = std::make_shared<core::MeshData>();
            int offset = 0;
            size_t index_offset = 0, material_offset = 0, texture_offset = 0;
            for (auto& m : meshes) {
                verts.push_back(m->vertices);
                idx.push_back(m->indices + offset);
                offset += m->vertex_count();
                for (auto material : m->materials) {
                    auto adjust = [&](uint32_t& texture) {
                        if (texture)
                            texture += static_cast<uint32_t>(texture_offset);
                    };
                    adjust(material.albedo_tex);
                    adjust(material.normal_tex);
                    adjust(material.metallic_roughness_tex);
                    adjust(material.emissive_tex);
                    adjust(material.ao_tex);
                    mesh->materials.push_back(std::move(material));
                }
                if (m->materials.empty())
                    mesh->materials.emplace_back();
                mesh->texture_images.insert(mesh->texture_images.end(), m->texture_images.begin(),
                                            m->texture_images.end());
                if (m->submeshes.empty() && m->indices.numel())
                    mesh->submeshes.push_back({index_offset, m->indices.numel(), material_offset});
                else
                    for (auto sub : m->submeshes) {
                        sub.start_index += index_offset;
                        sub.material_index += material_offset;
                        mesh->submeshes.push_back(sub);
                    }
                index_offset += m->indices.numel();
                material_offset += std::max<size_t>(1, m->materials.size());
                texture_offset += m->texture_images.size();
            }
            mesh->vertices = core::Tensor::cat(verts, 0);
            mesh->indices = core::Tensor::cat(idx, 0);
            const auto join_vertex_data = [&](Tensor core::MeshData::*member, size_t channels,
                                              float fallback) {
                if (!std::ranges::any_of(meshes, [&](const auto& source) {
                        return (source.get()->*member).is_valid();
                    }))
                    return Tensor{};
                std::vector<Tensor> parts;
                for (const auto& source : meshes) {
                    const auto& value = source.get()->*member;
                    parts.push_back(
                        value.is_valid()
                            ? value
                            : Tensor::full({static_cast<size_t>(source->vertex_count()), channels}, fallback,
                                           source->vertices.device()));
                }
                return Tensor::cat(parts, 0);
            };
            mesh->normals = join_vertex_data(&core::MeshData::normals, 3, 0);
            mesh->tangents = join_vertex_data(&core::MeshData::tangents, 4, 0);
            mesh->texcoords = join_vertex_data(&core::MeshData::texcoords, 2, 0);
            mesh->colors = join_vertex_data(&core::MeshData::colors, 4, 1);
            const auto device = mesh->vertices.device();
            MeshComponent component{std::move(mesh)};
            for (const auto* value : mesh_components)
                component.textures.insert(component.textures.end(), value->textures.begin(), value->textures.end());
            component.attributes = join_attributes(mesh_components, device);
            result.mesh = std::move(component);
        }
        return result;
    }

    void evaluate_join(NodeContext& context) {
        std::vector<Geometry> values;
        for (auto& v : context.inputs("Geometry"))
            if (auto* g = v.get_if<Geometry>())
                values.push_back(*g);
        context.set_output("Geometry", join_geometries(values));
    }

    Tensor integer_range(size_t count, Device device) {
        return (Tensor::ones({count}, device, DataType::Int32).cumsum(0) - 1).to(DataType::Int32);
    }

    Tensor selected_component_labels(const Tensor& positions, const Tensor& selected, float radius) {
        return core::radius_connected_components(positions, radius, selected.to(DataType::Bool));
    }

    Tensor cluster_average(const Tensor& values, const Tensor& labels, const Tensor& leaders) {
        const size_t count = labels.numel();
        auto sums = Tensor::zeros(values.shape(), values.device(), values.dtype());
        sums.index_add_(0, labels, values);
        auto counts = Tensor::zeros({count}, values.device(), DataType::Float32);
        counts.index_add_(0, labels, Tensor::ones({count}, values.device()));
        auto divisor = counts.maximum(1);
        while (divisor.ndim() < values.ndim())
            divisor = divisor.unsqueeze(1);
        return (sums / divisor).index_select(0, leaders);
    }

    void evaluate_merge_by_distance(NodeContext& context) {
        auto geometry = geometry_input(context);
        const float radius = input_float(context, "Distance", 0.01f);
        if (radius <= 0) {
            context.set_output("Geometry", std::move(geometry));
            return;
        }
        if (geometry.points) {
            auto& points = *geometry.points;
            const auto selected = selection(context, "Selection", field_context(points), true);
            const auto labels = selected_component_labels(points.positions, selected, radius);
            const auto leaders = labels.eq(integer_range(labels.numel(), labels.device())).nonzero().reshape({-1}).to(DataType::Int32);
            points.positions = cluster_average(points.positions, labels, leaders);
            points.colors = cluster_average(points.colors, labels, leaders);
            for (auto& [_, attribute] : points.attributes) {
                if (attribute.dtype() == DataType::Float32)
                    attribute = cluster_average(attribute, labels, leaders);
                else
                    attribute = attribute.index_select(0, leaders);
            }
        }
        if (geometry.mesh && geometry.mesh->mesh) {
            auto& component = *geometry.mesh;
            const auto& source = *component.mesh;
            const auto selected = selection(context, "Selection", field_context(component), true);
            const auto labels = selected_component_labels(source.vertices, selected, radius);
            const auto indices = integer_range(labels.numel(), labels.device());
            const auto leader_mask = labels.eq(indices);
            const auto leaders = leader_mask.nonzero().reshape({-1}).to(DataType::Int32);
            const auto compact = leader_mask.to(DataType::Int32).cumsum(0) - 1;
            const auto remap = compact.index_select(0, labels).to(DataType::Int32);
            auto faces = remap.index_select(0, source.indices.reshape({-1})).reshape(source.indices.shape());
            const auto a = faces.slice(1, 0, 1).squeeze(1);
            const auto b = faces.slice(1, 1, 2).squeeze(1);
            const auto c = faces.slice(1, 2, 3).squeeze(1);
            const auto keep = a.ne(b).logical_and(a.ne(c)).logical_and(b.ne(c));
            faces = faces.index_select(0, keep.nonzero().reshape({-1}).to(DataType::Int32)).contiguous();
            auto mesh = copy_mesh(source, cluster_average(source.vertices, labels, leaders), faces);
            if (source.has_colors())
                mesh->colors = cluster_average(source.colors, labels, leaders);
            if (source.has_normals())
                mesh->normals = normalized(cluster_average(source.normals, labels, leaders));
            if (source.has_tangents())
                mesh->tangents = cluster_average(source.tangents, labels, leaders);
            if (source.has_texcoords())
                mesh->texcoords = cluster_average(source.texcoords, labels, leaders);
            mesh->submeshes.clear();
            if (faces.numel())
                mesh->submeshes.push_back({0, faces.numel(), 0});
            component.mesh = std::move(mesh);
            for (auto& [_, attribute] : component.attributes) {
                if (attribute.dtype() == DataType::Float32)
                    attribute = cluster_average(attribute, labels, leaders);
                else
                    attribute = attribute.index_select(0, leaders);
            }
        }
        if (geometry.splats) {
            auto& splats = *geometry.splats;
            const auto selected = selection(context, "Selection", field_context(splats), true);
            const size_t count = splats.means.shape()[0];
            const auto opacity_order = splats.opacity.sort(0, true).second.to(DataType::Int32);
            auto ranks = Tensor::zeros({count}, splats.means.device(), DataType::Int32);
            ranks.scatter_(0, opacity_order, integer_range(count, splats.means.device()));
            // In opacity order, each component's smallest index is its most opaque splat.
            const auto labels = core::radius_connected_components(splats.means.index_select(0, opacity_order), radius,
                                                                  selected.to(DataType::Bool).index_select(0, opacity_order));
            const auto leaders = labels.eq(integer_range(count, splats.means.device())).index_select(0, ranks);
            const auto keep = selected.logical_not().logical_or(leaders);
            splats = filter_splats(splats, keep);
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void register_geometry(NodeTypeRegistry& registry) {
        const auto geo = std::string(GEOMETRY_SOCKET);
        const auto f = std::string(FLOAT_SOCKET);
        const auto v = std::string(VECTOR_SOCKET);
        register_type(registry, type("lfs.transform_geometry", "Geometry",
                                     geometry_inputs({in("Translation", v, glm::vec3(0)).step_size(0.01),
                                                      in("Rotation", v, glm::vec3(0)).step_size(1),
                                                      in("Scale", f, 1.0f).step_size(0.01)}),
                                     {out("Geometry", geo)}, evaluate_transform));
        register_type(registry, type("lfs.set_position", "Geometry",
                                     geometry_inputs({in("Selection", f, 1.0f, true)
                                                          .range(0, 1)
                                                          .step_size(0.01),
                                                      in("Position", v, glm::vec3(0), true).step_size(0.01),
                                                      in("Offset", v, glm::vec3(0), true).step_size(0.01)}),
                                     {out("Geometry", geo)}, evaluate_set_position));
        register_type(registry, type("lfs.delete_geometry", "Geometry",
                                     geometry_inputs(
                                         {in("Selection", f, 1.0f, true).range(0, 1).step_size(0.01)}),
                                     {out("Geometry", geo)}, evaluate_delete));
        register_type(registry,
                      type("lfs.separate_geometry", "Geometry",
                           geometry_inputs(
                               {in("Selection", f, 1.0f, true).range(0, 1).step_size(0.01)}),
                           {out("Selection", geo), out("Inverted", geo)}, evaluate_separate_geometry));
        register_type(registry,
                      type("lfs.join_geometry", "Geometry",
                           {in("Geometry", geo, {}, false, true)}, {out("Geometry", geo)}, evaluate_join));
        register_type(registry, type("lfs.merge_by_distance", "Geometry",
                                     geometry_inputs({in("Selection", f, 1.0f, true).range(0, 1).step_size(0.01),
                                                      in("Distance", f, 0.01f).minimum(0).step_size(0.001)}),
                                     {out("Geometry", geo)}, evaluate_merge_by_distance));
    }

} // namespace lfs::nodes::builtin

namespace lfs::nodes {
    std::shared_ptr<core::MeshData> transform_mesh(const core::MeshData& source, const glm::mat4& matrix) {
        using namespace builtin;
        const glm::mat3 linear(matrix);
        const auto device = source.vertices.device();
        auto mesh = copy_mesh(source,
                              source.vertices.matmul(matrix_tensor(linear, device)) +
                                  vector_tensor(glm::vec3(matrix[3]), device),
                              source.indices);
        const bool mirrors = glm::determinant(linear) < 0.0f;
        if (source.has_normals()) {
            // The cofactor matrix is the inverse transpose scaled by the determinant, and stays defined
            // when the transform flattens the mesh.
            glm::mat3 cofactor(glm::cross(linear[1], linear[2]), glm::cross(linear[2], linear[0]),
                               glm::cross(linear[0], linear[1]));
            if (mirrors)
                cofactor = -cofactor;
            mesh->normals = normalized(source.normals.matmul(matrix_tensor(cofactor, device)));
        }
        if (source.has_tangents()) {
            const auto directions = normalized(source.tangents.slice(1, 0, 3).matmul(matrix_tensor(linear, device)));
            auto handedness = source.tangents.slice(1, 3, 4);
            if (mirrors)
                handedness = handedness.neg();
            mesh->tangents = Tensor::cat({directions, handedness}, 1);
        }
        return mesh;
    }
} // namespace lfs::nodes
