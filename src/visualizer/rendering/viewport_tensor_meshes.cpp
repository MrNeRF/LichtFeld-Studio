/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "viewport_tensor_meshes.hpp"
#include "viewport_geometry.hpp"

#include "core/guarded_task.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/tensor.hpp"
#include "mesh_offscreen_renderer.hpp"
#include "rendering/render_constants.hpp"
#include "tensor_frame_uploads.hpp"
#include "viewport_depth_program.hpp"
#include "viewport_mesh_program.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <span>
#include <unordered_map>
#include <vector>

namespace lfs::vis {
    namespace {
        using Module = lfs::core::GpuKernelModule;
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;
        using Record = std::array<float, 4>;

        // Matches viewport_mesh.slang's Parameters.
        struct alignas(8) MeshParameters {
            std::uint64_t vertices = 0;
            std::uint64_t indices = 0;
            std::uint64_t uniforms = 0;
            std::uint64_t albedo = 0;
            std::uint64_t normal_map = 0;
            std::uint64_t metallic_roughness = 0;
            std::uint64_t shadow = 0;
            std::uint32_t first_index = 0;
            std::uint32_t padding = 0;
        };
        static_assert(sizeof(MeshParameters) == 64);

        // Matches viewport_depth.slang's Parameters.
        struct DepthParameters {
            std::uint64_t splat = 0;
            std::uint64_t depth = 0;
            std::uint32_t width = 0, height = 0;
            std::int32_t viewport_x = 0, viewport_y = 0;
            std::uint32_t viewport_width = 0, viewport_height = 0;
            std::uint32_t splat_width = 0, splat_height = 0;
            float near_plane = 0, far_plane = 0;
            std::uint32_t has_splat = 0, orthographic = 0;
        };
        static_assert(sizeof(DepthParameters) == 64);

        // Uniform record layout of viewport_mesh.slang.
        constexpr std::size_t kRecords = 23;
        enum : std::size_t {
            kMvp = 0,
            kModel = 4,
            kLightVp = 8,
            kCamera = 12,
            kLightDir = 13,
            kLight = 14,
            kSelection = 15,
            kBaseColor = 16,
            kEmissiveMetallic = 17,
            kRoughnessFlags = 18,
            kVertexColors = 19,
            kTextureSizes = 20,
            kTextureSizes2 = 21,
            kWireColor = 22,
        };
        using Uniforms = std::array<Record, kRecords>;

        void putMatrix(Uniforms& uniforms, const std::size_t at, const glm::mat4& matrix) {
            for (int column = 0; column < 4; ++column)
                uniforms[at + column] = {matrix[column].x, matrix[column].y, matrix[column].z, matrix[column].w};
        }

        struct Texture {
            Tensor image; // RGBA8 [H,W,4]
            std::array<float, 2> size{1, 1};
        };

        struct Material {
            Record base_color{1, 1, 1, 1};
            Record emissive_metallic{};
            Record roughness_flags{1, 0, 0, 0};
            Record vertex_colors{};
            Texture albedo, normal, metallic_roughness;
        };

        struct Submesh {
            std::uint32_t first_index = 0;
            std::uint32_t index_count = 0;
            std::size_t material = 0;
        };

        struct Shadow {
            int resolution = 0;
            Tensor color, depth;
            glm::mat4 light_vp{1.0f};
            glm::mat4 model{1.0f};
            glm::vec3 light_dir{0.0f};
            bool valid = false;
        };

        struct Mesh {
            std::uint32_t generation = 0;
            Tensor vertices; // [V*4, 4] float records
            Tensor indices;  // [F*3] int32
            std::uint32_t index_count = 0;
            glm::vec3 aabb_min{0.0f}, aabb_max{0.0f};
            std::vector<Material> materials;
            std::vector<Submesh> submeshes;
            Shadow shadow;
            std::uint64_t last_frame = 0;
        };

    } // namespace

    struct TensorMeshPass::Impl {
        std::unique_ptr<Module> mesh_program;
        std::unique_ptr<Module> depth_program;
        lfs::core::GpuBackend program_backend{};
        bool failed = false;
        Tensor white;        // 1x1 RGBA8 for absent textures
        Tensor dummy_shadow; // 1 float
        Tensor depth;        // [H,W] attachment
        TensorFrameUploads offscreen_uploads;
        std::unordered_map<std::uint64_t, Mesh> meshes;
        std::uint64_t frame = 0;

        // Programs and helper tensors live on the backend of the target.
        bool ensurePrograms(const lfs::core::GpuBackend backend) {
            if (mesh_program && program_backend == backend)
                return true;
            if (failed)
                return false;
            auto mesh = Module::load(viewport_mesh_program_entries(), backend);
            auto seed = Module::load(viewport_depth_program_entries(), backend);
            if (!mesh || !seed || !(*mesh)->supports_raster()) {
                LOG_ERROR("Could not load tensor mesh programs: {}",
                          !mesh ? mesh.error().detail() : !seed ? seed.error().detail()
                                                                : "raster unsupported");
                failed = true;
                return false;
            }
            const lfs::core::GpuBackendScope scope(backend);
            mesh_program = std::move(*mesh);
            depth_program = std::move(*seed);
            program_backend = backend;
            white = Tensor::full({1, 1, 4}, 255, Device::GPU, DataType::UInt8);
            dummy_shadow = Tensor::ones({1}, Device::GPU, DataType::Float32);
            depth = {};
            meshes.clear();
            return true;
        }

        static Texture uploadTexture(const lfs::core::MeshData& mesh, const std::uint32_t index,
                                     TensorFrameUploads& uploads) {
            if (index == 0 || index > mesh.texture_images.size())
                return {};
            const auto& image = mesh.texture_images[index - 1];
            if (image.pixels.empty() || image.width <= 0 || image.height <= 0)
                return {};
            const std::size_t pixels = std::size_t(image.width) * image.height;
            std::vector<std::uint8_t> rgba(pixels * 4);
            const int channels = image.channels;
            for (std::size_t i = 0; i < pixels; ++i) {
                const auto* source = image.pixels.data() + i * channels;
                auto* target = rgba.data() + i * 4;
                target[0] = channels >= 1 ? source[0] : 255;
                target[1] = channels >= 2 ? source[1] : target[0];
                target[2] = channels >= 3 ? source[2] : target[0];
                target[3] = channels >= 4 ? source[3] : 255;
            }
            return {uploads.upload(std::as_bytes(std::span(rgba)),
                                   {std::size_t(image.height), std::size_t(image.width), 4}, DataType::UInt8),
                    {float(image.width), float(image.height)}};
        }

        // Same attribute defaults as SharedViewportGpuAssets::uploadMesh.
        Mesh* prepareMesh(const lfs::core::MeshData& data, TensorFrameUploads& uploads) {
            auto& mesh = meshes[data.id()];
            mesh.last_frame = frame;
            if (mesh.vertices.is_valid() && mesh.generation == data.generation())
                return &mesh;
            const auto vertex_count = data.vertex_count();
            const auto face_count = data.face_count();
            if (vertex_count <= 0 || face_count <= 0)
                return nullptr;
            const auto positions = data.vertices.cpu().contiguous();
            const auto normals = data.has_normals() ? data.normals.cpu().contiguous() : Tensor{};
            const auto tangents = data.has_tangents() ? data.tangents.cpu().contiguous() : Tensor{};
            const auto texcoords = data.has_texcoords() ? data.texcoords.cpu().contiguous() : Tensor{};
            const auto colors = data.has_colors() ? data.colors.cpu().contiguous() : Tensor{};
            const float* pos = positions.ptr<float>();
            std::vector<Record> records(std::size_t(vertex_count) * 4);
            mesh.aabb_min = glm::vec3(std::numeric_limits<float>::max());
            mesh.aabb_max = glm::vec3(std::numeric_limits<float>::lowest());
            for (std::int64_t i = 0; i < vertex_count; ++i) {
                const glm::vec3 p{pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]};
                mesh.aabb_min = glm::min(mesh.aabb_min, p);
                mesh.aabb_max = glm::max(mesh.aabb_max, p);
                const float u = texcoords.is_valid() ? texcoords.ptr<float>()[i * 2] : 0.0f;
                const float v = texcoords.is_valid() ? texcoords.ptr<float>()[i * 2 + 1] : 0.0f;
                auto* out = &records[std::size_t(i) * 4];
                out[0] = {p.x, p.y, p.z, u};
                out[1] = normals.is_valid()
                             ? Record{normals.ptr<float>()[i * 3], normals.ptr<float>()[i * 3 + 1], normals.ptr<float>()[i * 3 + 2], v}
                             : Record{0.0f, 1.0f, 0.0f, v};
                out[2] = tangents.is_valid()
                             ? Record{tangents.ptr<float>()[i * 4], tangents.ptr<float>()[i * 4 + 1], tangents.ptr<float>()[i * 4 + 2], tangents.ptr<float>()[i * 4 + 3]}
                             : Record{0.0f, 0.0f, 0.0f, 1.0f};
                out[3] = colors.is_valid()
                             ? Record{colors.ptr<float>()[i * 4], colors.ptr<float>()[i * 4 + 1], colors.ptr<float>()[i * 4 + 2], colors.ptr<float>()[i * 4 + 3]}
                             : Record{1.0f, 1.0f, 1.0f, 1.0f};
            }
            const auto indices = data.indices.cpu().contiguous();
            mesh.index_count = static_cast<std::uint32_t>(face_count * 3);
            mesh.vertices = uploads.upload(std::as_bytes(std::span(records)), {records.size(), 4}, DataType::Float32);
            mesh.indices = uploads.upload(
                std::as_bytes(std::span(indices.ptr<std::int32_t>(), mesh.index_count)), {mesh.index_count},
                DataType::Int32);
            mesh.materials.clear();
            const std::size_t material_count = std::max<std::size_t>(data.materials.size(), 1);
            for (std::size_t m = 0; m < material_count; ++m) {
                const auto source = m < data.materials.size() ? data.materials[m] : lfs::core::Material{};
                Material material;
                material.albedo = uploadTexture(data, source.albedo_tex, uploads);
                material.normal = uploadTexture(data, source.normal_tex, uploads);
                material.metallic_roughness = uploadTexture(data, source.metallic_roughness_tex, uploads);
                material.base_color = {source.base_color.r, source.base_color.g, source.base_color.b, source.base_color.a};
                material.emissive_metallic = {source.emissive.r, source.emissive.g, source.emissive.b, source.metallic};
                material.roughness_flags = {source.roughness, material.albedo.image.is_valid() ? 1.0f : 0.0f,
                                            material.normal.image.is_valid() ? 1.0f : 0.0f,
                                            material.metallic_roughness.image.is_valid() ? 1.0f : 0.0f};
                material.vertex_colors = {data.has_colors() ? 1.0f : 0.0f, 0, 0, 0};
                mesh.materials.push_back(std::move(material));
            }
            mesh.submeshes.clear();
            for (const auto& submesh : data.submeshes)
                if (submesh.index_count > 0 && submesh.start_index + submesh.index_count <= mesh.index_count)
                    mesh.submeshes.push_back({static_cast<std::uint32_t>(submesh.start_index),
                                              static_cast<std::uint32_t>(submesh.index_count),
                                              std::min(submesh.material_index, mesh.materials.size() - 1)});
            if (mesh.submeshes.empty())
                mesh.submeshes.push_back({0, mesh.index_count, 0});
            mesh.generation = data.generation();
            mesh.shadow.valid = false;
            return &mesh;
        }

        void renderShadow(Mesh& mesh, const ViewportMeshDrawItem& item, TensorFrameUploads& uploads) {
            const int resolution = std::clamp(item.shadow_map_resolution, 256, 8192);
            auto& shadow = mesh.shadow;
            if (shadow.valid && shadow.resolution == resolution && shadow.model == item.model &&
                shadow.light_dir == item.light_dir)
                return;
            if (shadow.resolution != resolution || !shadow.depth.is_valid()) {
                shadow.color = Tensor::empty({std::size_t(resolution), std::size_t(resolution), 4}, Device::GPU, DataType::UInt8);
                shadow.depth = Tensor::empty({std::size_t(resolution), std::size_t(resolution)}, Device::GPU, DataType::Float32);
                shadow.resolution = resolution;
            }
            shadow.light_vp = meshShadowViewProjection(mesh.aabb_min, mesh.aabb_max, item.model, item.light_dir);
            Uniforms uniforms{};
            putMatrix(uniforms, kMvp, shadow.light_vp * item.model);
            auto uniform_tensor = uploads.upload(std::as_bytes(std::span(uniforms)), {kRecords, 4}, DataType::Float32);
            MeshParameters parameters;
            const std::array bindings{
                Module::Binding{0, &mesh.vertices}, Module::Binding{8, &mesh.indices},
                Module::Binding{16, &uniform_tensor}, Module::Binding{24, &white},
                Module::Binding{32, &white}, Module::Binding{40, &white},
                Module::Binding{48, &dummy_shadow}};
            // Front-face culling against acne, like the Vulkan shadow pipeline.
            auto result = mesh_program->draw({.vertex = "shadowVertex",
                                              .fragment = "shadowFragment",
                                              .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                                              .color = &shadow.color,
                                              .depth = &shadow.depth,
                                              .vertex_count = mesh.index_count,
                                              .cull = Module::Cull::Front,
                                              .depth_compare = Module::Compare::LessEqual,
                                              .depth_write = true,
                                              .clear_color = true,
                                              .clear_depth = true,
                                              .depth_clear = 1.0f});
            if (!result) {
                LOG_ERROR("Tensor mesh shadow pass failed: {}", result.error().detail());
                return;
            }
            shadow.model = item.model;
            shadow.light_dir = item.light_dir;
            shadow.valid = true;
        }

        void seedDepth(const ViewportFrameDesc& desc, const Module::Scissor& rect, const Tensor& destination) {
            const std::size_t height = destination.size(0), width = destination.size(1);
            if (!depth.is_valid() || depth.size(0) != height || depth.size(1) != width)
                depth = Tensor::empty({height, width}, Device::GPU, DataType::Float32);
            const auto& splat = desc.depth_blit.depth;
            const bool has_splat = splat && splat->is_valid() && splat->ndim() == 2 && splat->device() == Device::GPU;
            DepthParameters parameters{
                .width = std::uint32_t(width),
                .height = std::uint32_t(height),
                .viewport_x = std::int32_t(rect.x),
                .viewport_y = std::int32_t(rect.y),
                .viewport_width = rect.width,
                .viewport_height = rect.height,
                .splat_width = has_splat ? std::uint32_t(splat->size(1)) : 1u,
                .splat_height = has_splat ? std::uint32_t(splat->size(0)) : 1u,
                .near_plane = lfs::rendering::DEFAULT_NEAR_PLANE,
                .far_plane = lfs::rendering::DEFAULT_FAR_PLANE,
                .has_splat = has_splat ? 1u : 0u,
                // glm::ortho leaves w == 1.
                .orthographic = desc.mesh_view_projection[3][3] == 1.0f ? 1u : 0u,
            };
            const std::array bindings{Module::Binding{0, has_splat ? splat.get() : &dummy_shadow},
                                      Module::Binding{8, &depth, Module::Access::ReadWrite}};
            auto result = depth_program->dispatch({.function = "seedDepth",
                                                   .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                                                   .groups = {Module::groups_for(width, 64), std::uint32_t(height), 1}});
            if (!result)
                LOG_ERROR("Tensor mesh depth seed failed: {}", result.error().detail());
        }

        [[nodiscard]] lfs::Status record(Tensor& destination, Tensor& depth_attachment,
                                         const ViewportFrameDesc& desc,
                                         const Module::Scissor& rect,
                                         TensorFrameUploads& uploads,
                                         const bool seed_splat_depth) {
            ++frame;
            if (desc.mesh_items.empty() || rect.width == 0 || rect.height == 0)
                return {};
            const auto backend = lfs::core::gpu_backend_of(destination);
            if (!backend || !ensurePrograms(*backend))
                return {};
            const lfs::core::GpuBackendScope scope(*backend);
            if (seed_splat_depth)
                seedDepth(desc, rect, destination);

            struct Draw {
                std::string_view vertex, fragment;
                Mesh* mesh = nullptr;
                const Material* material = nullptr;
                std::uint32_t first_index = 0, index_count = 0;
                Module::Cull cull = Module::Cull::None;
                bool wire = false;
                bool shadow = false;
            };
            std::vector<Draw> draws;
            std::vector<Uniforms> uniforms;
            for (const auto& item : desc.mesh_items) {
                if (!item.mesh)
                    continue;
                Mesh* mesh = prepareMesh(*item.mesh, uploads);
                if (!mesh)
                    continue;
                if (item.shadow_enabled)
                    renderShadow(*mesh, item, uploads);
                const bool shadow = item.shadow_enabled && mesh->shadow.valid;
                Uniforms base{};
                putMatrix(base, kMvp, desc.mesh_view_projection * item.model);
                putMatrix(base, kModel, item.model);
                putMatrix(base, kLightVp, shadow ? mesh->shadow.light_vp : glm::mat4(1.0f));
                base[kCamera] = {desc.mesh_camera_position.x, desc.mesh_camera_position.y, desc.mesh_camera_position.z, 0};
                base[kLightDir] = {item.light_dir.x, item.light_dir.y, item.light_dir.z, 0};
                base[kLight] = {item.light_intensity, item.ambient, shadow ? 1.0f : 0.0f,
                                shadow ? float(mesh->shadow.resolution) : 1.0f};
                base[kSelection] = {item.is_emphasized ? 1.0f : 0.0f, item.dim_non_emphasized ? 1.0f : 0.0f,
                                    0.0f, item.wireframe_width};
                base[kWireColor] = {item.wireframe_color.r, item.wireframe_color.g, item.wireframe_color.b, 1};
                for (const auto& submesh : mesh->submeshes) {
                    const auto& material = mesh->materials[submesh.material];
                    Uniforms record = base;
                    record[kBaseColor] = material.base_color;
                    record[kEmissiveMetallic] = material.emissive_metallic;
                    record[kRoughnessFlags] = material.roughness_flags;
                    record[kVertexColors] = material.vertex_colors;
                    record[kTextureSizes] = {material.albedo.size[0], material.albedo.size[1],
                                             material.normal.size[0], material.normal.size[1]};
                    record[kTextureSizes2] = {material.metallic_roughness.size[0], material.metallic_roughness.size[1], 0, 0};
                    uniforms.push_back(record);
                    draws.push_back({"meshVertex", "meshFragment", mesh, &material, submesh.first_index,
                                     submesh.index_count,
                                     item.backface_culling ? Module::Cull::Back : Module::Cull::None, false, shadow});
                }
                if (item.wireframe_overlay) {
                    uniforms.push_back(base);
                    draws.push_back({"wireVertex", "wireFragment", mesh, nullptr, 0, mesh->index_count,
                                     Module::Cull::None, true, false});
                }
            }
            if (draws.empty())
                return {};
            const auto all_uniforms = uploads.upload(std::as_bytes(std::span(uniforms)),
                                                     {uniforms.size() * kRecords, 4}, DataType::Float32);
            std::deque<Tensor> slices;
            std::vector<MeshParameters> parameters(draws.size());
            std::vector<std::array<Module::Binding, 7>> bindings(draws.size());
            std::vector<Module::Draw> raster;
            raster.reserve(draws.size());
            const Module::Viewport viewport{float(rect.x), float(rect.y), float(rect.width), float(rect.height)};
            for (std::size_t i = 0; i < draws.size(); ++i) {
                const auto& draw = draws[i];
                const auto& slice = slices.emplace_back(all_uniforms.slice(0, i * kRecords, (i + 1) * kRecords));
                parameters[i].first_index = draw.first_index;
                const auto texture = [&](const Texture* t) { return t && t->image.is_valid() ? &t->image : &white; };
                bindings[i] = {Module::Binding{0, &draw.mesh->vertices}, Module::Binding{8, &draw.mesh->indices},
                               Module::Binding{16, &slice},
                               Module::Binding{24, texture(draw.material ? &draw.material->albedo : nullptr)},
                               Module::Binding{32, texture(draw.material ? &draw.material->normal : nullptr)},
                               Module::Binding{40, texture(draw.material ? &draw.material->metallic_roughness : nullptr)},
                               Module::Binding{48, draw.shadow ? &draw.mesh->shadow.depth : &dummy_shadow}};
                raster.push_back({.vertex = draw.vertex,
                                  .fragment = draw.fragment,
                                  .arguments = {std::as_bytes(std::span(&parameters[i], 1)), bindings[i]},
                                  .color = &destination,
                                  .depth = &depth_attachment,
                                  .vertex_count = draw.index_count,
                                  .scissor = rect,
                                  .viewport = viewport,
                                  .blend = draw.wire ? Module::Blend::StraightAlpha : Module::Blend::Opaque,
                                  .cull = draw.cull,
                                  .depth_compare = Module::Compare::LessEqual,
                                  .depth_write = !draw.wire});
            }
            auto result = mesh_program->draw_batch(raster);
            if (!result) {
                LOG_ERROR("Tensor mesh pass ({} draws) failed: {}", raster.size(), result.error().detail());
                return lfs::Status::failure(std::move(result.error()));
            }
            // Meshes not drawn for a while release their GPU copies.
            std::erase_if(meshes, [&](const auto& entry) { return frame - entry.second.last_frame > 120; });
            return {};
        }

        [[nodiscard]] lfs::Result<lfs::rendering::MeshLayer> renderOffscreen(
            const ViewportMeshPassDesc& mesh_desc, const glm::mat4& projection,
            const int width, const int height) {
            if (width <= 0 || height <= 0) {
                return lfs::make_error({
                    .code = lfs::ErrorCode::InvalidArgument,
                    .domain = lfs::ErrorDomain::Rendering,
                    .detail = "Mesh offscreen dimensions must be positive",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                });
            }

            Tensor color = Tensor::zeros(
                {static_cast<std::size_t>(height), static_cast<std::size_t>(width), 4},
                Device::GPU, DataType::Float32);
            Tensor device_depth = Tensor::full(
                {static_cast<std::size_t>(height), static_cast<std::size_t>(width)},
                1.0f, Device::GPU, DataType::Float32);
            ViewportFrameDesc frame;
            frame.framebuffer_extent = {width, height};
            frame.mesh_view_projection = mesh_desc.view_projection;
            frame.mesh_camera_position = mesh_desc.camera_position;
            frame.mesh_items = mesh_desc.items;
            const Module::Scissor rect{0, 0, static_cast<std::uint32_t>(width),
                                       static_cast<std::uint32_t>(height)};
            if (auto status = record(color, device_depth, frame, rect,
                                     offscreen_uploads, false);
                !status) {
                return std::move(status).error();
            }

            const Tensor host_color = color.cpu().contiguous();
            const Tensor host_depth = device_depth.cpu().contiguous();
            const std::size_t count = static_cast<std::size_t>(width) * height;
            std::vector<float> rgba(4 * count);
            std::vector<float> view_depth(count);
            const float* source_color = host_color.ptr<float>();
            const float* source_depth = host_depth.ptr<float>();
            for (std::size_t pixel = 0; pixel < count; ++pixel) {
                const float z_ndc = source_depth[pixel];
                rgba[pixel] = source_color[4 * pixel];
                rgba[count + pixel] = source_color[4 * pixel + 1];
                rgba[2 * count + pixel] = source_color[4 * pixel + 2];
                if (z_ndc >= 1.0f) {
                    rgba[3 * count + pixel] = 0.0f;
                    view_depth[pixel] = std::numeric_limits<float>::infinity();
                } else {
                    rgba[3 * count + pixel] = 1.0f;
                    view_depth[pixel] = linearizeMeshViewDepth(z_ndc, projection);
                }
            }
            return lfs::rendering::MeshLayer{
                .rgba = Tensor::from_vector(
                    rgba, {4, static_cast<std::size_t>(height), static_cast<std::size_t>(width)},
                    Device::CPU),
                .view_depth = Tensor::from_vector(
                    view_depth, {static_cast<std::size_t>(height), static_cast<std::size_t>(width)},
                    Device::CPU),
            };
        }
    };

    TensorMeshPass::TensorMeshPass() : impl_(std::make_unique<Impl>()) {}
    TensorMeshPass::~TensorMeshPass() = default;

    void TensorMeshPass::record(Tensor& destination, const ViewportFrameDesc& desc,
                                const Module::Scissor& rect, TensorFrameUploads& uploads) {
        try {
            // seedDepth sizes the splat-seeded depth attachment.
            static_cast<void>(impl_->record(destination, impl_->depth, desc, rect, uploads, true));
        } catch (const std::exception& error) {
            LOG_ERROR("Tensor mesh pass failed: {}", error.what());
        }
    }

    lfs::Result<lfs::rendering::MeshLayer> TensorMeshPass::renderOffscreen(
        const ViewportMeshPassDesc& desc, const glm::mat4& projection,
        const int width, const int height) {
        try {
            return impl_->renderOffscreen(desc, projection, width, height);
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): translate tensor program exceptions at the typed facade boundary.
            return lfs::core::detail::task_failure_from_current_exception<lfs::rendering::MeshLayer>(
                {.name = "mesh.tensor.offscreen", .domain = lfs::ErrorDomain::Rendering, .site = LFS_SOURCE_SITE_CURRENT()});
        }
    }
} // namespace lfs::vis
