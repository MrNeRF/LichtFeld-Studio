/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision_serialization.hpp"
#include "core/mesh_data.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "io/project_document.hpp"
#include "io/scene_chapter_adapter.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace lfs::core;
using namespace lfs::io::project;
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: bench_collision_project trained.licht volume.json output.licht\n";
        return 2;
    }
    try {
        if (std::filesystem::exists(argv[3]))
            throw std::runtime_error("Output project already exists; the trained project is never overwritten");
        auto check = [](const auto& value) {
            if (!value)
                throw std::runtime_error(lfs::format_for_developer(value.error()));
        };
        auto document = ProjectDocument::open(argv[1], {.defer_geometry_payloads = true});
        check(document);
        const auto original = document->scene_graph().nodes();
        check(original);
        nlohmann::json data;
        std::ifstream(argv[2]) >> data;
        auto volume = std::make_shared<CollisionVolume>(data.get<CollisionVolume>());
        volume->show_voxels = true;
        auto bounds = volume->boundsMesh();
        auto mesh = std::make_shared<MeshData>();
        mesh->vertices = Tensor::from_vector(bounds.vertices, {bounds.vertices.size() / 3, 3}, Device::CPU);
        mesh->indices = Tensor::from_vector(bounds.indices, {bounds.indices.size() / 3, 3}, Device::CPU);
        Scene scene;
        const auto id = scene.addMesh("Collision", mesh);
        auto* node = scene.getNodeById(id);
        const auto uuid = node->uuid;
        node->collision = volume;
        node->collision_display = {.preview = true, .sharp = true, .surface_ready = false, .opacity = .4f};
        node->local_transform = glm::mat4{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1};
        node->payload_diverged = true;
        ScenePayloadBindings bindings{{uuid, {.fourcc = "MESH", .instance_uuid = uuid, .source_kind = "mesh"}}};
        auto graph = capture_scene_graph(scene, bindings);
        check(graph);
        auto added = graph->nodes();
        check(added);
        auto record = added->front();
        for (const auto& old : *original)
            if (!old.parent_uuid)
                record.child_order = std::max(record.child_order, old.child_order + 1);
        check(document->edit_scene_graph().upsert_node(record));
        check(document->set_mesh(uuid, MeshPayload(mesh)));
        check(document->edit_project().upsert_embedded_payload_provenance({
            .uuid = uuid,
            .node_uuid = uuid,
            .fourcc = "MESH",
            .import_locator = {.preferred = "generated:" + uuid.to_string(), .base = LocatorBase::Absolute},
            .import_fingerprint = {},
            .content_xxh3_128 = {},
        }));
        check(document->edit_project().upsert_embed_decision({
            .uuid = uuid,
            .node_uuid = uuid,
            .payload_fourcc = "MESH",
            .decision = "embedded",
            .reference_uuid = std::nullopt,
            .reason = "project-owned scene payload",
        }));
        ProjectDocumentSaveOptions options;
        options.file_uuid = generate_uuid_v4();
        options.save_as_project_uuid = generate_uuid_v4();
        options.commit.commit_uuid = generate_uuid_v4();
        const auto checkpoint = document->bound_checkpoint_uuid();
        check(checkpoint);
        options.commit.snapshot_uuid = checkpoint->value_or(generate_uuid_v4());
        check(document->save_as(argv[3], options));
        auto reopened = ProjectDocument::open(argv[3], {.defer_geometry_payloads = true});
        check(reopened);
        auto nodes = reopened->scene_graph().nodes();
        check(nodes);
        if (nodes->size() != original->size() + 1)
            throw std::runtime_error("Archive node count changed unexpectedly");
        for (const auto& old : *original) {
            const auto found = std::find_if(nodes->begin(), nodes->end(), [&](const auto& n) { return n.uuid == old.uuid; });
            if (found == nodes->end() || !(*found == old))
                throw std::runtime_error("Original scene node or camera changed");
        }
        Scene restored;
        auto shell = stage_scene_shell(reopened->scene_graph(), restored);
        check(shell);
        auto retired = restored.commitRestoreStage(std::move(*shell));
        auto plan = reopened->stage_hydration(restored);
        check(plan);
        const auto report = ProjectDocument::commit_partial_hydration(restored, std::move(*plan), true);
        if (report.invalidated_payload_units)
            throw std::runtime_error("A project payload failed hydration");
        const auto* collider = restored.getNodeByUuid(uuid);
        if (!collider || !collider->mesh || !collider->collision || *collider->collision != *volume)
            throw std::runtime_error("Collision payload did not hydrate exactly");
        size_t gaussians = 0, cameras = 0;
        int sh_degree = 0;
        for (const auto* n : restored.getNodes()) {
            if (n->model) {
                gaussians += size_t(n->model->size());
                sh_degree = std::max(sh_degree, n->model->get_active_sh_degree());
            }
            cameras += n->type == NodeType::CAMERA;
        }
        std::cout << nlohmann::json{{"project", argv[3]}, {"gaussians", gaussians}, {"sh_degree", sh_degree}, {"cameras", cameras}, {"original_nodes_unchanged", original->size()}, {"occupied", volume->occupiedCount()}, {"hydrated_units", report.hydrated_payload_units}, {"collision_uuid", uuid.to_string()}}.dump() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
