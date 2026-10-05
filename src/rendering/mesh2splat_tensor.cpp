/* Derived from Mesh2Splat by Electronic Arts Inc.
 * Original: Copyright (c) 2025 Electronic Arts Inc. All rights reserved.
 * Licensed under BSD 3-Clause (see THIRD_PARTY_LICENSES.md)
 *
 * Modifications: Copyright (c) 2025-2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mesh2splat_program.hpp"
#include "mesh2splat_shared.hpp"
#include "rendering/mesh2splat.hpp"

#include "core/gpu_kernel_module.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstring>
#include <format>
#include <mutex>
#include <span>
#include <vector>

namespace lfs::rendering {

    using core::DataType;
    using core::Device;
    using core::GpuBackend;
    using core::GpuKernelModule;
    using core::Mesh2SplatOptions;
    using core::Mesh2SplatProgressCallback;
    using core::MeshData;
    using core::SplatData;
    using core::Tensor;
    using core::TextureImage;

    namespace {

        using mesh2splat_detail::conversion_error;
        using mesh2splat_detail::GaussianVertex;

        // mesh2splat.slang Parameters: the Vulkan push constants plus the
        // tensor addresses and texture sizes that replace its descriptors.
        struct Parameters {
            uint64_t pointers[6] = {};
            glm::vec4 material_factor{1.0f};
            glm::vec4 bbox_min_metallic{0.0f};
            glm::vec4 bbox_max_roughness{0.0f};
            glm::uvec4 flags{0};
            glm::uvec2 texture_sizes[3] = {};
            uint32_t capacity = 0;
            uint32_t padding = 0;
        };
        static_assert(sizeof(Parameters) == 144);

        struct Texture {
            Tensor texels; // RGBA8, invalid when the material has no such texture
            glm::uvec2 size{0};
        };

        struct MaterialTextures {
            Texture albedo;
            Texture normal;
            Texture metallic_roughness;
            bool albedo_srgb = false;
        };

        [[nodiscard]] lfs::Result<Texture> upload_texture(const TextureImage& img) {
            auto rgba = mesh2splat_detail::to_rgba8(img);
            if (!rgba)
                return std::move(rgba).error();
            return Texture{Tensor::from_blob(rgba->data(), {rgba->size()}, Device::CPU, DataType::UInt8).to(Device::GPU),
                           glm::uvec2(img.width, img.height)};
        }

        // Same texture selection as the Vulkan converter's upload_material_textures.
        [[nodiscard]] lfs::Result<std::vector<MaterialTextures>> upload_material_textures(const MeshData& mesh) {
            std::vector<MaterialTextures> textures(mesh.materials.size());
            const auto image = [&](const bool present, const uint32_t index) -> const TextureImage* {
                if (!present || index == 0 || index > mesh.texture_images.size())
                    return nullptr;
                const auto& img = mesh.texture_images[index - 1];
                return img.pixels.empty() ? nullptr : &img;
            };
            for (size_t i = 0; i < mesh.materials.size(); ++i) {
                const auto& mat = mesh.materials[i];
                const std::array<std::pair<const TextureImage*, Texture*>, 3> slots{{
                    {image(mat.has_albedo_texture(), mat.albedo_tex), &textures[i].albedo},
                    {image(mat.has_normal_texture(), mat.normal_tex), &textures[i].normal},
                    {image(mat.has_metallic_roughness_texture(), mat.metallic_roughness_tex), &textures[i].metallic_roughness},
                }};
                for (const auto& [img, texture] : slots) {
                    if (!img)
                        continue;
                    auto uploaded = upload_texture(*img);
                    if (!uploaded)
                        return std::move(uploaded).error();
                    *texture = std::move(*uploaded);
                }
                // The Vulkan converter samples albedo through an sRGB view when it has color channels.
                textures[i].albedo_srgb = slots[0].first && slots[0].first->channels >= 3;
            }
            return textures;
        }

    } // namespace

    lfs::Result<std::unique_ptr<SplatData>>
    mesh_to_splat_tensor(const MeshData& mesh,
                         const Mesh2SplatOptions& options,
                         const GpuBackend backend,
                         Mesh2SplatProgressCallback progress) {
        static std::mutex conversion_mutex;
        std::lock_guard conversion_lock(conversion_mutex);
        const core::GpuBackendScope backend_scope(backend);

        auto report = [&](float pct, const std::string& stage) -> bool {
            return progress ? progress(pct, stage) : true;
        };

        auto prepared = mesh2splat_detail::prepare_conversion(mesh, options, progress);
        if (!prepared)
            return std::move(prepared).error();
        const auto& submesh_geometries = prepared->submeshes;
        const int res = options.resolution_target;

        LOG_INFO("mesh2splat: {} tensor converter, {} submeshes, {} triangles, resolution={}",
                 core::gpu_backend_name(backend), submesh_geometries.size(), prepared->triangle_count, res);

        if (!report(0.15f, "Loading tensor program"))
            return conversion_error(ErrorCode::Cancelled, "Cancelled");

        auto loaded = GpuKernelModule::load(mesh2splat_program_entries(), backend);
        if (!loaded)
            return std::move(loaded).error();
        const auto module = std::move(*loaded);
        if (!module->supports_raster())
            return conversion_error(ErrorCode::Unsupported,
                                    std::format("Mesh2Splat needs raster support, which the {} tensor backend lacks",
                                                core::gpu_backend_name(backend)));

        auto material_textures = upload_material_textures(mesh);
        if (!material_textures)
            return std::move(material_textures).error();

        Tensor output = Tensor::empty({size_t{prepared->output_capacity}, sizeof(GaussianVertex) / sizeof(float)}, Device::GPU);
        Tensor counter = Tensor::zeros({1}, Device::GPU, DataType::UInt32);
        // The fragments write splats, not pixels; like the Vulkan framebuffer the
        // attachment only defines the res x res raster grid.
        Tensor attachment = Tensor::empty({static_cast<size_t>(res), static_cast<size_t>(res), 4}, Device::GPU, DataType::UInt8);

        std::vector<Tensor> vertex_buffers;
        vertex_buffers.reserve(submesh_geometries.size());
        for (const auto& geo : submesh_geometries) {
            vertex_buffers.push_back(Tensor::from_blob(const_cast<mesh2splat_detail::PerVertexData*>(geo.vertices.data()),
                                                       {geo.vertices.size(), sizeof(mesh2splat_detail::PerVertexData) / sizeof(float)},
                                                       Device::CPU, DataType::Float32)
                                         .to(Device::GPU));
        }

        using Binding = GpuKernelModule::Binding;
        constexpr auto RW = GpuKernelModule::Access::ReadWrite;
        std::vector<Parameters> parameters(submesh_geometries.size());
        std::vector<std::array<Binding, 6>> bindings(submesh_geometries.size());
        std::vector<GpuKernelModule::Draw> draws(submesh_geometries.size());
        for (size_t i = 0; i < submesh_geometries.size(); ++i) {
            const auto& geo = submesh_geometries[i];
            Parameters& pc = parameters[i];
            pc.bbox_min_metallic = glm::vec4(prepared->global_min, 0.0f);
            pc.bbox_max_roughness = glm::vec4(prepared->global_max, 1.0f);
            pc.flags.w = mesh.has_colors() ? 1 : 0;
            const MaterialTextures* mt = nullptr;
            if (geo.material_index < mesh.materials.size()) {
                const auto& mat = mesh.materials[geo.material_index];
                pc.material_factor = mat.base_color;
                pc.bbox_min_metallic.w = mat.metallic;
                pc.bbox_max_roughness.w = mat.roughness;
                mt = &(*material_textures)[geo.material_index];
                pc.flags.x = mt->albedo.texels.is_valid() ? (mt->albedo_srgb ? 2 : 1) : 0;
                pc.flags.y = mt->normal.texels.is_valid() ? 1 : 0;
                pc.flags.z = mt->metallic_roughness.texels.is_valid() ? 1 : 0;
                pc.texture_sizes[0] = mt->albedo.size;
                pc.texture_sizes[1] = mt->normal.size;
                pc.texture_sizes[2] = mt->metallic_roughness.size;
            }
            pc.capacity = prepared->output_capacity;

            const auto texels = [](const Texture* texture) {
                return texture && texture->texels.is_valid() ? &texture->texels : nullptr;
            };
            bindings[i] = {Binding{0, &vertex_buffers[i]},
                           Binding{8, &output, RW},
                           Binding{16, &counter, RW},
                           Binding{24, texels(mt ? &mt->albedo : nullptr)},
                           Binding{32, texels(mt ? &mt->normal : nullptr)},
                           Binding{40, texels(mt ? &mt->metallic_roughness : nullptr)}};
            draws[i] = {.vertex = "mesh2splatVertex",
                        .fragment = "mesh2splatFragment",
                        .arguments = {std::as_bytes(std::span(&pc, 1)), bindings[i]},
                        .color = &attachment,
                        .vertex_count = static_cast<uint32_t>(geo.vertices.size()),
                        .clear_color = i == 0};
        }

        if (!report(0.3f, "Converting mesh to splats"))
            return conversion_error(ErrorCode::Cancelled, "Cancelled");

        if (auto drawn = module->draw_batch(draws); !drawn)
            return std::move(drawn).error();

        uint32_t num_gaussians = counter.to(Device::CPU).ptr<uint32_t>()[0];
        if (num_gaussians == 0)
            return conversion_error(ErrorCode::Internal, "Conversion produced zero gaussians");
        if (num_gaussians > prepared->output_capacity) {
            LOG_WARN("mesh2splat: atomic counter ({}) exceeds output capacity ({}), clamping",
                     num_gaussians, prepared->output_capacity);
            num_gaussians = prepared->output_capacity;
        }

        if (!report(0.85f, "Reading back data"))
            return conversion_error(ErrorCode::Cancelled, "Cancelled");

        const Tensor host_output = output.slice(0, 0, num_gaussians).contiguous().to(Device::CPU);
        std::vector<GaussianVertex> gpu_data(num_gaussians);
        std::memcpy(gpu_data.data(), host_output.ptr<float>(), gpu_data.size() * sizeof(GaussianVertex));

        if (!report(0.9f, "Building SplatData"))
            return conversion_error(ErrorCode::Cancelled, "Cancelled");

        LOG_INFO("mesh2splat: produced {} gaussians (resolution={})", num_gaussians, res);
        auto splat = mesh2splat_detail::build_splat_data(gpu_data, options.sigma / static_cast<float>(res),
                                                         prepared->scene_scale);

        if (!report(1.0f, "Complete"))
            return conversion_error(ErrorCode::Cancelled, "Cancelled");
        return splat;
    }

#ifdef LFS_MESH2SPLAT_TENSOR_DEFAULT
    // Builds without the Vulkan converter (no Vulkan, or Apple: MoltenVK has no
    // geometry shaders) keep the legacy string-error entry point.
    std::expected<std::unique_ptr<SplatData>, std::string>
    mesh_to_splat(const MeshData& mesh,
                  const Mesh2SplatOptions& options,
                  Mesh2SplatProgressCallback progress) {
        auto splat = mesh_to_splat_tensor(mesh, options, core::default_gpu_backend(), std::move(progress));
        if (!splat)
            return std::unexpected(std::string(splat.error().detail()));
        return std::move(*splat);
    }
#endif

} // namespace lfs::rendering
