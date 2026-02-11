/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mesh_loader.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/path_utils.hpp"
#include "io/error.hpp"
#include "io/mesh/texture_loader.hpp"
#include <array>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <cassert>
#include <chrono>
#include <format>
#include <limits>

namespace lfs::io {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Material;
    using lfs::core::MeshData;
    using lfs::core::Tensor;

    constexpr std::array MESH_EXTENSIONS = {".obj", ".fbx", ".gltf", ".glb", ".stl", ".dae", ".3ds", ".ply"};

    namespace {

        Material extract_material(const aiMaterial* ai_mat) {
            Material mat;

            aiColor4D color;
            if (ai_mat->Get(AI_MATKEY_BASE_COLOR, color) == AI_SUCCESS ||
                ai_mat->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS) {
                mat.base_color = {color.r, color.g, color.b, color.a};
            }

            aiColor3D emissive;
            if (ai_mat->Get(AI_MATKEY_COLOR_EMISSIVE, emissive) == AI_SUCCESS) {
                aiString emissive_tex_path;
                bool has_emissive_tex = ai_mat->GetTexture(aiTextureType_EMISSIVE, 0, &emissive_tex_path) == AI_SUCCESS;
                if (!has_emissive_tex) {
                    mat.emissive = {emissive.r, emissive.g, emissive.b};
                }
            }

            ai_mat->Get(AI_MATKEY_METALLIC_FACTOR, mat.metallic);
            ai_mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, mat.roughness);

            int two_sided = 0;
            if (ai_mat->Get(AI_MATKEY_TWOSIDED, two_sided) == AI_SUCCESS) {
                mat.double_sided = (two_sided != 0);
            }

            aiString ai_name;
            if (ai_mat->Get(AI_MATKEY_NAME, ai_name) == AI_SUCCESS) {
                mat.name = ai_name.C_Str();
            }

            aiString tex_path;
            if (ai_mat->GetTexture(aiTextureType_BASE_COLOR, 0, &tex_path) == AI_SUCCESS ||
                ai_mat->GetTexture(aiTextureType_DIFFUSE, 0, &tex_path) == AI_SUCCESS) {
                mat.albedo_tex_path = tex_path.C_Str();
            }

            if (ai_mat->GetTexture(aiTextureType_NORMALS, 0, &tex_path) == AI_SUCCESS) {
                mat.normal_tex_path = tex_path.C_Str();
            }

            if (ai_mat->GetTexture(aiTextureType_UNKNOWN, 0, &tex_path) == AI_SUCCESS ||
                ai_mat->GetTexture(aiTextureType_METALNESS, 0, &tex_path) == AI_SUCCESS) {
                mat.metallic_roughness_tex_path = tex_path.C_Str();
            }

            return mat;
        }

        MeshData convert_ai_mesh(const aiMesh* ai_mesh) {
            assert(ai_mesh);
            const int nv = static_cast<int>(ai_mesh->mNumVertices);
            assert(nv > 0);

            auto vertices = Tensor::empty({static_cast<size_t>(nv), size_t{3}}, Device::CPU, DataType::Float32);
            auto vacc = vertices.accessor<float, 2>();
            for (int i = 0; i < nv; ++i) {
                const auto& v = ai_mesh->mVertices[i];
                vacc(i, 0) = v.x;
                vacc(i, 1) = v.y;
                vacc(i, 2) = v.z;
            }

            const int nf = static_cast<int>(ai_mesh->mNumFaces);
            int tri_count = 0;
            for (int i = 0; i < nf; ++i) {
                if (ai_mesh->mFaces[i].mNumIndices == 3)
                    ++tri_count;
            }
            if (tri_count < nf) {
                LOG_WARN("Mesh has {} non-triangle faces (skipped), {} triangles kept",
                         nf - tri_count, tri_count);
            }

            auto indices = Tensor::empty({static_cast<size_t>(tri_count), size_t{3}}, Device::CPU, DataType::Int32);
            auto iacc = indices.accessor<int32_t, 2>();
            int ti = 0;
            for (int i = 0; i < nf; ++i) {
                const auto& face = ai_mesh->mFaces[i];
                if (face.mNumIndices != 3)
                    continue;
                iacc(ti, 0) = static_cast<int32_t>(face.mIndices[0]);
                iacc(ti, 1) = static_cast<int32_t>(face.mIndices[1]);
                iacc(ti, 2) = static_cast<int32_t>(face.mIndices[2]);
                ++ti;
            }
            assert(ti == tri_count);

            MeshData mesh(std::move(vertices), std::move(indices));

            if (ai_mesh->HasNormals()) {
                mesh.normals = Tensor::empty({static_cast<size_t>(nv), size_t{3}}, Device::CPU, DataType::Float32);
                auto nacc = mesh.normals.accessor<float, 2>();
                for (int i = 0; i < nv; ++i) {
                    const auto& n = ai_mesh->mNormals[i];
                    nacc(i, 0) = n.x;
                    nacc(i, 1) = n.y;
                    nacc(i, 2) = n.z;
                }
            }

            if (ai_mesh->HasTangentsAndBitangents()) {
                mesh.tangents = Tensor::empty({static_cast<size_t>(nv), size_t{4}}, Device::CPU, DataType::Float32);
                auto tacc = mesh.tangents.accessor<float, 2>();
                for (int i = 0; i < nv; ++i) {
                    const auto& t = ai_mesh->mTangents[i];
                    const auto& b = ai_mesh->mBitangents[i];
                    const auto& n = ai_mesh->mNormals[i];
                    const float handedness = (n.x * (t.y * b.z - t.z * b.y) -
                                              n.y * (t.x * b.z - t.z * b.x) +
                                              n.z * (t.x * b.y - t.y * b.x)) < 0.0f
                                                 ? -1.0f
                                                 : 1.0f;
                    tacc(i, 0) = t.x;
                    tacc(i, 1) = t.y;
                    tacc(i, 2) = t.z;
                    tacc(i, 3) = handedness;
                }
            }

            if (ai_mesh->HasTextureCoords(0)) {
                mesh.texcoords = Tensor::empty({static_cast<size_t>(nv), size_t{2}}, Device::CPU, DataType::Float32);
                auto tcacc = mesh.texcoords.accessor<float, 2>();
                for (int i = 0; i < nv; ++i) {
                    const auto& tc = ai_mesh->mTextureCoords[0][i];
                    tcacc(i, 0) = tc.x;
                    tcacc(i, 1) = tc.y;
                }
            }

            if (ai_mesh->HasVertexColors(0)) {
                mesh.colors = Tensor::empty({static_cast<size_t>(nv), size_t{4}}, Device::CPU, DataType::Float32);
                auto cacc = mesh.colors.accessor<float, 2>();
                for (int i = 0; i < nv; ++i) {
                    const auto& c = ai_mesh->mColors[0][i];
                    cacc(i, 0) = c.r;
                    cacc(i, 1) = c.g;
                    cacc(i, 2) = c.b;
                    cacc(i, 3) = c.a;
                }
            }

            return mesh;
        }

        MeshData merge_meshes(std::vector<MeshData> meshes, const std::vector<unsigned int>& material_indices) {
            assert(!meshes.empty());
            assert(meshes.size() == material_indices.size());
            if (meshes.size() == 1) {
                auto& m = meshes[0];
                if (m.submeshes.empty()) {
                    m.submeshes.push_back({0, static_cast<size_t>(m.face_count() * 3), material_indices[0]});
                }
                return std::move(m);
            }

            int total_verts = 0;
            int total_faces = 0;
            bool any_normals = false;
            bool any_tangents = false;
            bool any_texcoords = false;
            bool any_colors = false;

            for (const auto& m : meshes) {
                total_verts += static_cast<int>(m.vertex_count());
                total_faces += static_cast<int>(m.face_count());
                any_normals |= m.has_normals();
                any_tangents |= m.has_tangents();
                any_texcoords |= m.has_texcoords();
                any_colors |= m.has_colors();
            }

            auto vertices = Tensor::empty({static_cast<size_t>(total_verts), size_t{3}}, Device::CPU, DataType::Float32);
            auto indices = Tensor::empty({static_cast<size_t>(total_faces), size_t{3}}, Device::CPU, DataType::Int32);

            Tensor normals, tangents, texcoords, colors;
            if (any_normals)
                normals = Tensor::zeros({static_cast<size_t>(total_verts), size_t{3}}, Device::CPU, DataType::Float32);
            if (any_tangents)
                tangents = Tensor::zeros({static_cast<size_t>(total_verts), size_t{4}}, Device::CPU, DataType::Float32);
            if (any_texcoords)
                texcoords = Tensor::zeros({static_cast<size_t>(total_verts), size_t{2}}, Device::CPU, DataType::Float32);
            if (any_colors)
                colors = Tensor::ones({static_cast<size_t>(total_verts), size_t{4}}, Device::CPU, DataType::Float32);

            auto vacc = vertices.accessor<float, 2>();
            auto iacc = indices.accessor<int32_t, 2>();

            int v_offset = 0;
            int f_offset = 0;

            MeshData result;

            for (auto& m : meshes) {
                if (v_offset > std::numeric_limits<int32_t>::max()) {
                    LOG_ERROR("Vertex offset {} exceeds Int32 range, truncating merge", v_offset);
                    break;
                }
                const int nv = static_cast<int>(m.vertex_count());
                const int nf = static_cast<int>(m.face_count());

                auto src_vacc = m.vertices.accessor<float, 2>();
                for (int i = 0; i < nv; ++i) {
                    vacc(v_offset + i, 0) = src_vacc(i, 0);
                    vacc(v_offset + i, 1) = src_vacc(i, 1);
                    vacc(v_offset + i, 2) = src_vacc(i, 2);
                }

                auto src_iacc = m.indices.accessor<int32_t, 2>();
                for (int i = 0; i < nf; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        assert(src_iacc(i, j) >= 0 && src_iacc(i, j) < nv);
                    }
                    iacc(f_offset + i, 0) = src_iacc(i, 0) + static_cast<int32_t>(v_offset);
                    iacc(f_offset + i, 1) = src_iacc(i, 1) + static_cast<int32_t>(v_offset);
                    iacc(f_offset + i, 2) = src_iacc(i, 2) + static_cast<int32_t>(v_offset);
                }

                if (any_normals && m.has_normals()) {
                    auto dst = normals.accessor<float, 2>();
                    auto src = m.normals.accessor<float, 2>();
                    for (int i = 0; i < nv; ++i) {
                        dst(v_offset + i, 0) = src(i, 0);
                        dst(v_offset + i, 1) = src(i, 1);
                        dst(v_offset + i, 2) = src(i, 2);
                    }
                }

                if (any_tangents && m.has_tangents()) {
                    auto dst = tangents.accessor<float, 2>();
                    auto src = m.tangents.accessor<float, 2>();
                    for (int i = 0; i < nv; ++i) {
                        dst(v_offset + i, 0) = src(i, 0);
                        dst(v_offset + i, 1) = src(i, 1);
                        dst(v_offset + i, 2) = src(i, 2);
                        dst(v_offset + i, 3) = src(i, 3);
                    }
                }

                if (any_texcoords && m.has_texcoords()) {
                    auto dst = texcoords.accessor<float, 2>();
                    auto src = m.texcoords.accessor<float, 2>();
                    for (int i = 0; i < nv; ++i) {
                        dst(v_offset + i, 0) = src(i, 0);
                        dst(v_offset + i, 1) = src(i, 1);
                    }
                }

                if (any_colors && m.has_colors()) {
                    auto dst = colors.accessor<float, 2>();
                    auto src = m.colors.accessor<float, 2>();
                    for (int i = 0; i < nv; ++i) {
                        dst(v_offset + i, 0) = src(i, 0);
                        dst(v_offset + i, 1) = src(i, 1);
                        dst(v_offset + i, 2) = src(i, 2);
                        dst(v_offset + i, 3) = src(i, 3);
                    }
                }

                const size_t sub_idx = static_cast<size_t>(&m - &meshes[0]);
                result.submeshes.push_back({static_cast<size_t>(f_offset * 3),
                                            static_cast<size_t>(nf * 3),
                                            material_indices[sub_idx]});

                v_offset += nv;
                f_offset += nf;
            }

            result.vertices = std::move(vertices);
            result.indices = std::move(indices);
            if (any_normals)
                result.normals = std::move(normals);
            if (any_tangents)
                result.tangents = std::move(tangents);
            if (any_texcoords)
                result.texcoords = std::move(texcoords);
            if (any_colors)
                result.colors = std::move(colors);

            return result;
        }

    } // namespace

    Result<LoadResult> MeshLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        const auto start_time = std::chrono::high_resolution_clock::now();
        const auto path_str = lfs::core::path_to_utf8(path);

        if (options.progress) {
            options.progress(0.0f, "Loading mesh file...");
        }

        if (!std::filesystem::exists(path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND, "Mesh file does not exist", path);
        }

        constexpr unsigned int IMPORT_FLAGS =
            aiProcess_Triangulate |
            aiProcess_GenSmoothNormals |
            aiProcess_CalcTangentSpace |
            aiProcess_JoinIdenticalVertices |
            aiProcess_FlipUVs |
            aiProcess_SortByPType;

        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path_str, IMPORT_FLAGS);

        if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("Assimp failed: {}", importer.GetErrorString()), path);
        }

        if (options.progress) {
            options.progress(30.0f, "Processing meshes...");
        }

        LOG_INFO("Mesh file: {} meshes, {} materials, {} embedded textures",
                 scene->mNumMeshes, scene->mNumMaterials, scene->mNumTextures);

        std::vector<Material> materials;
        materials.reserve(scene->mNumMaterials);
        for (unsigned int i = 0; i < scene->mNumMaterials; ++i) {
            materials.push_back(extract_material(scene->mMaterials[i]));
            LOG_DEBUG("  material[{}]: name='{}', albedo_path='{}', normal_path='{}', mr_path='{}'",
                      i, materials.back().name, materials.back().albedo_tex_path,
                      materials.back().normal_tex_path, materials.back().metallic_roughness_tex_path);
        }

        std::vector<MeshData> sub_meshes;
        std::vector<unsigned int> mesh_material_indices;
        sub_meshes.reserve(scene->mNumMeshes);
        mesh_material_indices.reserve(scene->mNumMeshes);
        for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
            sub_meshes.push_back(convert_ai_mesh(scene->mMeshes[i]));
            mesh_material_indices.push_back(scene->mMeshes[i]->mMaterialIndex);
            LOG_DEBUG("  mesh[{}]: {} verts, {} faces, material_index={}",
                      i, scene->mMeshes[i]->mNumVertices, scene->mMeshes[i]->mNumFaces,
                      scene->mMeshes[i]->mMaterialIndex);
        }

        if (options.progress) {
            options.progress(50.0f, "Loading textures...");
        }

        const auto model_dir = path.parent_path();
        mesh::TextureLoader tex_loader;
        std::vector<lfs::core::TextureImage> texture_images;

        auto load_texture = [&](const std::string& tex_path) -> uint32_t {
            if (tex_path.empty())
                return 0;

            if (tex_path[0] == '*') {
                const unsigned int embed_idx = std::stoul(tex_path.substr(1));
                if (embed_idx < scene->mNumTextures) {
                    const auto* ai_tex = scene->mTextures[embed_idx];
                    auto td = tex_loader.load_from_memory(
                        reinterpret_cast<const uint8_t*>(ai_tex->pcData), ai_tex->mWidth);
                    if (!td.pixels.empty()) {
                        lfs::core::TextureImage img;
                        img.pixels = std::move(td.pixels);
                        img.width = td.width;
                        img.height = td.height;
                        img.channels = td.channels;
                        texture_images.push_back(std::move(img));
                        return static_cast<uint32_t>(texture_images.size());
                    }
                }
                return 0;
            }

            const auto full_path = model_dir / tex_path;
            const auto* td = tex_loader.load_from_file(full_path);
            if (td) {
                lfs::core::TextureImage img;
                img.pixels = td->pixels;
                img.width = td->width;
                img.height = td->height;
                img.channels = td->channels;
                texture_images.push_back(std::move(img));
                return static_cast<uint32_t>(texture_images.size());
            }
            return 0;
        };

        for (size_t mi = 0; mi < materials.size(); ++mi) {
            auto& mat = materials[mi];
            mat.albedo_tex = load_texture(mat.albedo_tex_path);
            mat.normal_tex = load_texture(mat.normal_tex_path);
            mat.metallic_roughness_tex = load_texture(mat.metallic_roughness_tex_path);
            LOG_DEBUG("  material[{}] textures: albedo={}, normal={}, mr={}",
                      mi, mat.albedo_tex, mat.normal_tex, mat.metallic_roughness_tex);
        }

        if (options.progress) {
            options.progress(70.0f, "Merging mesh data...");
        }

        auto mesh_data = std::make_shared<MeshData>(merge_meshes(std::move(sub_meshes), mesh_material_indices));
        mesh_data->materials = std::move(materials);
        mesh_data->texture_images = std::move(texture_images);

        if (!mesh_data->has_normals()) {
            mesh_data->compute_normals();
        }

        if (options.progress) {
            options.progress(100.0f, "Mesh loading complete");
        }

        const auto end_time = std::chrono::high_resolution_clock::now();
        const auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        LOG_INFO("Loaded mesh: {} vertices, {} faces in {}ms",
                 mesh_data->vertex_count(), mesh_data->face_count(), load_time.count());

        LoadResult result{
            .data = std::move(mesh_data),
            .scene_center = Tensor::zeros({3}, Device::CPU),
            .loader_used = name(),
            .load_time = load_time,
            .warnings = {}};

        return result;
    }

    bool MeshLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path) || std::filesystem::is_directory(path)) {
            return false;
        }

        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        for (const auto& e : MESH_EXTENSIONS) {
            if (ext == e)
                return true;
        }
        return false;
    }

    std::string MeshLoader::name() const {
        return "Mesh";
    }

    std::vector<std::string> MeshLoader::supportedExtensions() const {
        std::vector<std::string> result;
        result.reserve(MESH_EXTENSIONS.size() * 2);
        for (const auto& ext : MESH_EXTENSIONS) {
            result.emplace_back(ext);
            std::string upper(ext);
            std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
            result.push_back(std::move(upper));
        }
        return result;
    }

    int MeshLoader::priority() const {
        return 5;
    }

} // namespace lfs::io
