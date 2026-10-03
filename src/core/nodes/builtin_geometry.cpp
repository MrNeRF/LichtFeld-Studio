/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"
#include "core/splat_data_transform.hpp"
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
        if (geometry.splats) {
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
            geometry.mesh = MeshComponent{std::move(m), geometry.mesh->textures};
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void evaluate_transform(NodeContext& context) {
        auto geometry = geometry_input(context);
        const auto translation = input_vector(context, "Translation");
        const auto rotation = rotation_matrix(input_vector(context, "Rotation"));
        const float scale = input_float(context, "Scale", 1);
        const auto matrix =
            glm::translate(glm::mat4(1), translation) * rotation * glm::scale(glm::mat4(1), glm::vec3(scale));
        if (geometry.splats) {
            auto attributes = geometry.splats->attributes;
            auto data = splat_data_from_geometry(geometry);
            core::transform(*data, matrix);
            geometry.splats = geometry_from_splat_data(*data).splats;
            geometry.splats->attributes = std::move(attributes);
        }
        const auto positions = [&](const Tensor& value) {
            return value.matmul(matrix_tensor(glm::mat3(matrix), value.device())) +
                   vector_tensor(translation, value.device());
        };
        if (geometry.points)
            geometry.points->positions = positions(geometry.points->positions);
        if (geometry.mesh && geometry.mesh->mesh) {
            const auto& source = *geometry.mesh->mesh;
            auto mesh = copy_mesh(source, positions(source.vertices), source.indices);
            if (source.has_normals()) {
                const auto transformed = source.normals.matmul(
                    matrix_tensor(glm::transpose(glm::inverse(glm::mat3(matrix))), source.normals.device()));
                mesh->normals = safe_divide(transformed, (transformed * transformed).sum(1, true).sqrt());
            }
            geometry.mesh = MeshComponent{std::move(mesh), geometry.mesh->textures};
        }
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
            result.mesh = MeshComponent{filter_mesh_faces(*source.mesh->mesh, mask, selected), source.mesh->textures};
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
        for (auto& g : values) {
            if (g.splats)
                splats.push_back(&*g.splats);
            if (g.points)
                points.push_back(&*g.points);
            if (g.mesh && g.mesh->mesh)
                meshes.push_back(g.mesh->mesh);
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
            const auto join_vertex_data = [&](Tensor core::MeshData::* member, size_t channels,
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
            MeshComponent component{std::move(mesh)};
            for (const auto& value : values)
                if (value.mesh && value.mesh->mesh)
                    component.textures.insert(component.textures.end(), value.mesh->textures.begin(), value.mesh->textures.end());
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

    void register_geometry(NodeTypeRegistry& registry) {
        const auto geo = std::string(GEOMETRY_SOCKET);
        const auto f = std::string(FLOAT_SOCKET);
        const auto v = std::string(VECTOR_SOCKET);
        register_type(registry, type("lfs.transform_geometry", "Transform Geometry", "Geometry",
                                     "Translate, rotate and uniformly scale every geometry component.",
                                     geometry_inputs({in("Translation", v, glm::vec3(0)).step_size(0.01),
                                                      in("Rotation", v, glm::vec3(0)).step_size(1),
                                                      in("Scale", f, 1.0f).step_size(0.01)}),
                                     {out("Geometry", geo)}, evaluate_transform));
        register_type(registry, type("lfs.set_position", "Set Position", "Geometry",
                                     "Set element positions and add an offset, blended by selection.",
                                     geometry_inputs({in("Selection", f, 1.0f, true)
                                                          .range(0, 1)
                                                          .step_size(0.01),
                                                      in("Position", v, glm::vec3(0), true).step_size(0.01),
                                                      in("Offset", v, glm::vec3(0), true).step_size(0.01)}),
                                     {out("Geometry", geo)}, evaluate_set_position));
        register_type(registry, type("lfs.delete_geometry", "Delete Geometry", "Geometry",
                                     "Remove selected elements and mesh faces touching selected vertices.",
                                     geometry_inputs(
                                         {in("Selection", f, 1.0f, true).range(0, 1).step_size(0.01)}),
                                     {out("Geometry", geo)}, evaluate_delete));
        register_type(registry,
                      type("lfs.separate_geometry", "Separate Geometry", "Geometry",
                           "Split geometry into selected and unselected components.",
                           geometry_inputs(
                               {in("Selection", f, 1.0f, true).range(0, 1).step_size(0.01)}),
                           {out("Selection", geo), out("Inverted", geo)}, evaluate_separate_geometry));
        register_type(registry,
                      type("lfs.join_geometry", "Join Geometry", "Geometry",
                           "Concatenate geometry components and pad missing SH coefficients.",
                           {in("Geometry", geo, {}, false, true)}, {out("Geometry", geo)}, evaluate_join));
    }

} // namespace lfs::nodes::builtin
