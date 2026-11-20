/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "io/skinned_mesh.hpp"
#include <algorithm>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <format>
#include <limits>

namespace lfs::io {

    namespace {

        // Assimp matrices are stored by rows, glm ones by columns.
        [[nodiscard]] glm::mat4 to_glm(const aiMatrix4x4& m) {
            return {m.a1, m.b1, m.c1, m.d1,
                    m.a2, m.b2, m.c2, m.d2,
                    m.a3, m.b3, m.c3, m.d3,
                    m.a4, m.b4, m.c4, m.d4};
        }

    } // namespace

    Result<SkinnedMesh> load_skinned_mesh(const std::filesystem::path& path) {
        if (!lfs::core::native_fs::exists(path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND, "Skinned mesh file does not exist", path);
        }

        // Neither aiProcess_PreTransformVertices, which bakes the skeleton
        // away, nor aiProcess_LimitBoneWeights, which may drop the bones that
        // move no vertex: callers match bones by name and need all of them.
        constexpr unsigned int IMPORT_FLAGS =
            aiProcess_Triangulate |
            aiProcess_GenSmoothNormals |
            aiProcess_JoinIdenticalVertices |
            aiProcess_SortByPType;

        Assimp::Importer importer;
        const aiScene* const scene = importer.ReadFile(lfs::core::path_to_native_utf8(path), IMPORT_FLAGS);
        if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("Assimp failed: {}", importer.GetErrorString()), path);
        }

        const aiMesh* source = nullptr;
        for (unsigned int i = 0; i < scene->mNumMeshes && !source; ++i) {
            const aiMesh* const candidate = scene->mMeshes[i];
            if (candidate->HasBones() && candidate->HasPositions() && candidate->HasNormals() &&
                candidate->HasFaces() && (candidate->mPrimitiveTypes & aiPrimitiveType_TRIANGLE)) {
                source = candidate;
            }
        }
        if (!source) {
            return make_error(ErrorCode::UNSUPPORTED_FORMAT, "No triangle mesh with a skeleton in the file", path);
        }
        if (source->mNumBones > std::numeric_limits<std::uint16_t>::max()) {
            return make_error(ErrorCode::UNSUPPORTED_FORMAT, "Too many bones in the skeleton", path);
        }

        SkinnedMesh mesh;
        mesh.vertices.resize(source->mNumVertices);
        for (unsigned int i = 0; i < source->mNumVertices; ++i) {
            const aiVector3D& position = source->mVertices[i];
            const aiVector3D& normal = source->mNormals[i];
            mesh.vertices[i].position = {position.x, position.y, position.z};
            mesh.vertices[i].normal = {normal.x, normal.y, normal.z};
        }

        // Assimp lists the vertices of each bone, a renderer wants the bones
        // of each vertex: every vertex keeps its four heaviest.
        mesh.bones.reserve(source->mNumBones);
        for (unsigned int index = 0; index < source->mNumBones; ++index) {
            const aiBone* const bone = source->mBones[index];
            mesh.bones.push_back({bone->mName.C_Str(), to_glm(bone->mOffsetMatrix)});
            for (unsigned int i = 0; i < bone->mNumWeights; ++i) {
                const aiVertexWeight& weight = bone->mWeights[i];
                // A bone that moves no vertex comes with one weight of zero.
                if (!(weight.mWeight > 0.0f))
                    continue;
                if (weight.mVertexId >= mesh.vertices.size()) {
                    return make_error(ErrorCode::CORRUPTED_DATA, "Bone weight for a vertex the mesh does not have", path);
                }
                auto& vertex = mesh.vertices[weight.mVertexId];
                const auto lightest = std::min_element(vertex.weights.begin(), vertex.weights.end());
                if (weight.mWeight > *lightest) {
                    *lightest = weight.mWeight;
                    vertex.bones[static_cast<std::size_t>(lightest - vertex.weights.begin())] = static_cast<std::uint16_t>(index);
                }
            }
        }
        for (auto& vertex : mesh.vertices) {
            float total = 0.0f;
            for (const float weight : vertex.weights)
                total += weight;
            if (!(total > 1e-6f)) {
                return make_error(ErrorCode::CORRUPTED_DATA, "Vertex that no bone moves", path);
            }
            for (float& weight : vertex.weights)
                weight /= total;
        }

        mesh.indices.reserve(static_cast<std::size_t>(source->mNumFaces) * 3);
        for (unsigned int i = 0; i < source->mNumFaces; ++i) {
            const aiFace& face = source->mFaces[i];
            if (face.mNumIndices != 3)
                continue;
            for (unsigned int corner = 0; corner < 3; ++corner) {
                if (face.mIndices[corner] >= mesh.vertices.size()) {
                    return make_error(ErrorCode::CORRUPTED_DATA, "Triangle corner outside the mesh", path);
                }
                mesh.indices.push_back(face.mIndices[corner]);
            }
        }
        if (mesh.indices.empty()) {
            return make_error(ErrorCode::UNSUPPORTED_FORMAT, "No triangle in the skinned mesh", path);
        }
        return mesh;
    }

} // namespace lfs::io
