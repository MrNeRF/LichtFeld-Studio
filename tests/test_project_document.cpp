/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/loaders/loader_utils.hpp"
#include "io/project_document.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace {

    namespace fs = std::filesystem;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::NodeType;
    using lfs::core::Scene;
    using lfs::core::Tensor;
    using lfs::core::Uuid;
    using namespace lfs::io::project;

    Uuid fixed_uuid(const std::uint64_t tag) {
        const auto parsed = Uuid::from_string(
            std::format("70000000-0000-4000-8000-{:012x}", tag));
        EXPECT_TRUE(parsed);
        return parsed.value_or(Uuid{});
    }

    class TemporaryDirectory {
    public:
        TemporaryDirectory()
            : path(fs::temp_directory_path() /
                   ("lfs-p3-document-" +
                    lfs::core::generate_uuid_v4().to_string())) {
            fs::create_directories(path);
        }

        ~TemporaryDirectory() {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }

        fs::path path;
    };

    std::unique_ptr<lfs::core::SplatData>
    make_splat(const std::size_t count) {
        std::vector<float> means(count * 3, 0.0f);
        std::vector<float> rotations(count * 4, 0.0f);
        for (std::size_t index = 0; index < count; ++index) {
            means[index * 3] = static_cast<float>(index);
            rotations[index * 4] = 1.0f;
        }
        return std::make_unique<lfs::core::SplatData>(
            0,
            Tensor::from_vector(means, {count, std::size_t{3}}, Device::CPU),
            Tensor::zeros({count, std::size_t{1}, std::size_t{3}},
                          Device::CPU, DataType::Float32),
            Tensor{},
            Tensor::zeros({count, std::size_t{3}}, Device::CPU,
                          DataType::Float32),
            Tensor::from_vector(rotations, {count, std::size_t{4}},
                                Device::CPU),
            Tensor::zeros({count, std::size_t{1}}, Device::CPU,
                          DataType::Float32),
            1.0f);
    }

    std::shared_ptr<lfs::core::PointCloud>
    make_point_cloud(const std::size_t count) {
        std::vector<float> means(count * 3, 0.0f);
        std::vector<float> colors(count * 3, 0.5f);
        for (std::size_t index = 0; index < count; ++index) {
            means[index * 3] = static_cast<float>(index + 10);
        }
        return std::make_shared<lfs::core::PointCloud>(
            Tensor::from_vector(means, {count, std::size_t{3}}, Device::CPU),
            Tensor::from_vector(colors, {count, std::size_t{3}}, Device::CPU));
    }

    std::shared_ptr<lfs::core::MeshData> make_mesh() {
        return std::make_shared<lfs::core::MeshData>(
            Tensor::from_vector(
                std::vector<float>{
                    -1.0f,
                    -1.0f,
                    0.0f,
                    1.0f,
                    -1.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                    0.0f,
                },
                {std::size_t{3}, std::size_t{3}}, Device::CPU),
            Tensor::from_vector(
                std::vector<std::int32_t>{0, 1, 2},
                {std::size_t{1}, std::size_t{3}}, Device::CPU));
    }

    Tensor selection_tensor(
        const std::initializer_list<std::uint8_t> values) {
        auto result = Tensor::empty({values.size()}, Device::CPU,
                                    DataType::UInt8);
        std::ranges::copy(values, result.ptr<std::uint8_t>());
        return result;
    }

    ReferenceFingerprint fake_fingerprint(const std::uint8_t tag) {
        ReferenceFingerprint result;
        result.kind = FingerprintKind::File;
        result.size = 1000 + tag;
        result.mtime_unix_ns = 2000 + tag;
        result.head_xxh3.bytes.fill(tag);
        result.tail_xxh3.bytes.fill(
            static_cast<std::uint8_t>(tag + 1));
        return result;
    }

    EmbeddedPayloadProvenance provenance(
        const Uuid& node_uuid, const std::string& fourcc,
        const std::string& locator, const std::uint8_t tag) {
        return EmbeddedPayloadProvenance{
            .uuid = node_uuid,
            .node_uuid = node_uuid,
            .fourcc = fourcc,
            .import_locator =
                {.preferred = locator, .base = LocatorBase::Project},
            .import_fingerprint = fake_fingerprint(tag),
            .content_xxh3_128 = {},
        };
    }

    void require_status(lfs::Result<void> result) {
        ASSERT_TRUE(result)
            << (result ? std::string{}
                       : lfs::format_for_developer(result.error()));
    }

    ProjectDocumentSaveOptions save_options(
        const std::uint64_t identity_tag,
        const std::uint64_t wallclock) {
        return ProjectDocumentSaveOptions{
            .commit =
                {
                    .kind = CommitKind::Explicit,
                    .commit_uuid = fixed_uuid(identity_tag),
                    .snapshot_uuid = fixed_uuid(identity_tag + 1),
                    .wallclock_unix_ns = wallclock,
                },
            .file_uuid = fixed_uuid(identity_tag + 2),
            .index_compression =
                IndexCompression::StoredForDeterministicTests,
            .disk_reserve_bytes = 0,
        };
    }

    std::vector<std::byte> tensor_bytes(
        const Tensor& tensor) {
        if (!tensor.is_valid()) {
            return {};
        }
        const auto cpu = tensor.cpu().contiguous();
        std::vector<std::byte> result(cpu.bytes());
        if (!result.empty()) {
            std::memcpy(
                result.data(), cpu.data_ptr(), result.size());
        }
        return result;
    }

    struct LiveSceneWitness {
        std::vector<std::string> metadata;
        std::vector<std::vector<std::byte>> tensor_payloads;

        friend bool operator==(const LiveSceneWitness&,
                               const LiveSceneWitness&) = default;
    };

    LiveSceneWitness witness_scene(
        const Scene& scene) {
        LiveSceneWitness result;
        result.metadata.push_back(std::format(
            "training={}", scene.getTrainingModelNodeUuid().to_string()));
        for (const auto* node : scene.getNodes()) {
            std::string transform_bits;
            for (const float value :
                 std::span(
                     glm::value_ptr(
                         node->local_transform.get()),
                     std::size_t{16})) {
                transform_bits += std::format(
                    "{:08x}",
                    std::bit_cast<std::uint32_t>(value));
            }
            std::string children;
            for (const auto child : node->children) {
                children += std::format("{},", child);
            }
            result.metadata.push_back(std::format(
                "node={} id={} parent={} type={} name={} visible={} "
                "locked={} training={} diverged={} count={} ptr={} "
                "model={} point={} mesh={} children={} transform={}",
                node->uuid.to_string(), node->id,
                node->parent_id,
                static_cast<unsigned>(node->type), node->name,
                node->visible.get(), node->locked.get(),
                node->training_enabled, node->payload_diverged,
                node->gaussian_count.load(
                    std::memory_order_acquire),
                reinterpret_cast<std::uintptr_t>(node),
                reinterpret_cast<std::uintptr_t>(
                    node->model.get()),
                reinterpret_cast<std::uintptr_t>(
                    node->point_cloud.get()),
                reinterpret_cast<std::uintptr_t>(
                    node->mesh.get()),
                children, transform_bits));
            if (node->model) {
                result.tensor_payloads.push_back(
                    tensor_bytes(node->model->means_raw()));
            }
            if (node->point_cloud) {
                result.tensor_payloads.push_back(
                    tensor_bytes(node->point_cloud->means));
                result.tensor_payloads.push_back(
                    tensor_bytes(node->point_cloud->colors));
            }
            if (node->mesh) {
                result.tensor_payloads.push_back(
                    tensor_bytes(node->mesh->vertices));
                result.tensor_payloads.push_back(
                    tensor_bytes(node->mesh->indices));
            }
        }
        for (const auto& group :
             scene.getSelectionGroups()) {
            result.metadata.push_back(std::format(
                "group={} name={} color={:08x}{:08x}{:08x} "
                "count={} locked={}",
                group.id, group.name,
                std::bit_cast<std::uint32_t>(group.color.x),
                std::bit_cast<std::uint32_t>(group.color.y),
                std::bit_cast<std::uint32_t>(group.color.z),
                group.count, group.locked));
        }
        result.metadata.push_back(std::format(
            "active-group={}", scene.getActiveSelectionGroup()));
        for (const auto domain :
             {lfs::core::SelectionDomain::Splat,
              lfs::core::SelectionDomain::PointCloud}) {
            const auto mask =
                scene.getSelectionMask(domain);
            result.metadata.push_back(std::format(
                "mask-{}={}",
                static_cast<unsigned>(domain),
                mask ? "present" : "absent"));
            if (mask) {
                result.tensor_payloads.push_back(
                    tensor_bytes(*mask));
            }
        }
        return result;
    }

    void write_phase_a_fixture(const fs::path& path) {
        const Uuid project_uuid = fixed_uuid(950);
        const Uuid root_uuid = fixed_uuid(951);
        const Uuid splat_uuid = fixed_uuid(952);
        const Uuid point_uuid = fixed_uuid(953);
        const Uuid mesh_uuid = fixed_uuid(954);
        const Uuid reference_uuid = fixed_uuid(955);

        auto document =
            ProjectDocument::create(project_uuid, 100);
        ASSERT_TRUE(document)
            << lfs::format_for_developer(document.error());
        require_status(
            document->edit_references().upsert(
                ReferenceRecord{
                    .uuid = reference_uuid,
                    .key = "phase-a.reference",
                    .kind = "dataset",
                    .locator =
                        {
                            .preferred = "../dataset",
                            .base = LocatorBase::Project,
                        },
                    .fingerprint = fake_fingerprint(40),
                    .unresolved = true,
                }));
        auto& scene = document->edit_scene_graph();
        require_status(scene.upsert_node(SceneNodeRecord{
            .uuid = root_uuid,
            .type = "group",
            .name = "Root",
            .child_order = 0,
        }));
        require_status(scene.upsert_node(SceneNodeRecord{
            .uuid = splat_uuid,
            .type = "splat",
            .name = "Splat",
            .parent_uuid = root_uuid,
            .child_order = 0,
            .payload =
                PayloadBinding{
                    .fourcc = "SPLT",
                    .instance_uuid = splat_uuid,
                    .source_kind = "ply",
                },
        }));
        require_status(scene.upsert_node(SceneNodeRecord{
            .uuid = point_uuid,
            .type = "pointcloud",
            .name = "Points",
            .parent_uuid = root_uuid,
            .child_order = 1,
            .payload =
                PayloadBinding{
                    .fourcc = "PCLD",
                    .instance_uuid = point_uuid,
                    .source_kind = "ply",
                },
        }));
        require_status(scene.upsert_node(SceneNodeRecord{
            .uuid = mesh_uuid,
            .type = "mesh",
            .name = "Mesh",
            .parent_uuid = root_uuid,
            .child_order = 2,
            .payload =
                PayloadBinding{
                    .fourcc = "MESH",
                    .instance_uuid = mesh_uuid,
                    .source_kind = "obj",
                },
        }));

        auto splat = SplatChapterPayload::capture(
            *make_splat(3), SplatSourceKind::ImportedPly,
            false);
        ASSERT_TRUE(splat)
            << lfs::format_for_developer(splat.error());
        require_status(
            document->set_splat(
                splat_uuid, std::move(*splat)));
        require_status(document->set_point_cloud(
            point_uuid,
            PointCloudPayload(make_point_cloud(2))));
        require_status(document->set_mesh(
            mesh_uuid, MeshPayload(make_mesh())));
        auto& project = document->edit_project();
        for (const auto& [uuid, fourcc, tag] :
             std::array{
                 std::tuple{splat_uuid, std::string{"SPLT"},
                            std::uint8_t{41}},
                 std::tuple{point_uuid, std::string{"PCLD"},
                            std::uint8_t{42}},
                 std::tuple{mesh_uuid, std::string{"MESH"},
                            std::uint8_t{43}},
             }) {
            require_status(project.upsert_embed_decision(
                EmbedDecision{
                    .uuid = uuid,
                    .node_uuid = uuid,
                    .payload_fourcc = fourcc,
                    .decision = "embedded",
                    .reason = "Phase-A fixture",
                }));
            require_status(
                project
                    .upsert_embedded_payload_provenance(
                        provenance(
                            uuid, fourcc,
                            std::format(
                                "assets/{}.bin", fourcc),
                            tag)));
        }
        require_status(
            document->edit_selection().set_groups(
                {
                    lfs::core::SelectionGroup{
                        .id = 4,
                        .name = "Fixture",
                        .color = {0.2f, 0.3f, 0.4f},
                    },
                },
                4, 5));
        require_status(
            document->edit_selection().upsert_slice(
                SelectionMaskSlice{
                    .node_uuid = point_uuid,
                    .domain =
                        lfs::core::SelectionDomain::
                            PointCloud,
                    .mask = {4, 0},
                }));
        const auto saved =
            document->save(path, save_options(960, 200));
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(saved.error());
    }

    TEST(SplatChapterTest,
         ImportedBytesRoundTripAndTrainingOrLiveRadEmbeddingIsRefused) {
        auto model = make_splat(3);
        auto imported = SplatChapterPayload::capture(
            *model, SplatSourceKind::ImportedPly, false);
        ASSERT_TRUE(imported)
            << lfs::format_for_developer(imported.error());
        EXPECT_EQ(imported->lfsp_version(), 4u);

        auto reparsed =
            SplatChapterPayload::from_lfsp(imported->bytes());
        ASSERT_TRUE(reparsed)
            << lfs::format_for_developer(reparsed.error());
        EXPECT_TRUE(std::ranges::equal(reparsed->bytes(),
                                       imported->bytes()));
        auto hydrated = reparsed->hydrate();
        ASSERT_TRUE(hydrated)
            << lfs::format_for_developer(hydrated.error());
        EXPECT_EQ((*hydrated)->size(), 3u);

        auto training = SplatChapterPayload::capture(
            *model, SplatSourceKind::ImportedPly, true);
        EXPECT_FALSE(training);
        EXPECT_EQ(training.error().code(),
                  lfs::ErrorCode::FailedPrecondition);
        auto live_rad = SplatChapterPayload::capture(
            *model, SplatSourceKind::LiveRad, false);
        EXPECT_FALSE(live_rad);
        EXPECT_EQ(live_rad.error().code(),
                  lfs::ErrorCode::FailedPrecondition);
    }

    TEST(LoaderGeoreferenceTest,
         CentralizedOriginNeverRoundTripsThroughFloat) {
        const float lower = 1.0f;
        const float upper =
            std::nextafter(lower, 2.0f);
        auto points =
            std::make_shared<lfs::core::PointCloud>(
                Tensor::from_vector(
                    std::vector<float>{
                        lower,
                        0.0f,
                        0.0f,
                        upper,
                        0.0f,
                        0.0f,
                    },
                    {std::size_t{2}, std::size_t{3}},
                    Device::CPU),
                Tensor::zeros(
                    {std::size_t{2}, std::size_t{3}},
                    Device::CPU, DataType::Float32));
        std::vector<std::shared_ptr<lfs::core::Camera>>
            cameras;
        auto centralized = lfs::io::centralize_scene(
            cameras, points,
            lfs::io::CentralizeDataset::ByPointCloud,
            Tensor::zeros(
                {std::size_t{3}}, Device::CPU,
                DataType::Float32));
        ASSERT_TRUE(centralized.georeference);
        const double expected =
            (static_cast<double>(lower) +
             static_cast<double>(upper)) /
            2.0;
        EXPECT_DOUBLE_EQ(
            centralized.georeference->world_origin[0],
            expected);
        EXPECT_NE(
            centralized.georeference->world_origin[0],
            static_cast<double>(
                static_cast<float>(expected)));
    }

    TEST(SceneChapterAdapterTest,
         DuplicateNamesHydrateWithDistinctStableUuids) {
        SceneGraphChapter chapter;
        const Uuid first_uuid = fixed_uuid(900);
        const Uuid second_uuid = fixed_uuid(901);
        require_status(chapter.upsert_node(SceneNodeRecord{
            .uuid = first_uuid,
            .type = "group",
            .name = "Duplicate",
            .child_order = 0,
        }));
        require_status(chapter.upsert_node(SceneNodeRecord{
            .uuid = second_uuid,
            .type = "group",
            .name = "Duplicate",
            .child_order = 1,
        }));

        Scene live;
        ASSERT_NE(live.addGroup("Existing"), lfs::core::NULL_NODE);
        auto restored =
            hydrate_scene_graph(chapter, live, ScenePayloadResolver{});
        ASSERT_TRUE(restored)
            << lfs::format_for_developer(restored.error());
        EXPECT_EQ(live.getNodeCount(), 2u);
        ASSERT_NE(live.getNodeByUuid(first_uuid), nullptr);
        ASSERT_NE(live.getNodeByUuid(second_uuid), nullptr);
        EXPECT_EQ(live.getNodeByUuid(first_uuid)->name, "Duplicate");
        EXPECT_EQ(live.getNodeByUuid(second_uuid)->name, "Duplicate");
        ASSERT_EQ(
            live.getNodeIdByName("Duplicate"),
            live.getNodeIdByUuid(first_uuid));
        live.removeNodeById(live.getNodeIdByUuid(first_uuid));
        EXPECT_EQ(live.getNodeCount(), 1u);
        EXPECT_EQ(
            live.getNodeIdByName("Duplicate"),
            live.getNodeIdByUuid(second_uuid));
    }

    TEST(SceneChapterAdapterTest,
         MismatchedTypeSpecificStateRefusesHydrationBeforeClearingLiveScene) {
        SceneGraphChapter chapter;
        require_status(chapter.upsert_node(SceneNodeRecord{
            .uuid = fixed_uuid(902),
            .type = "group",
            .name = "Malformed group",
            .child_order = 0,
            .cropbox = CropBoxRecord{},
        }));

        Scene live;
        ASSERT_NE(live.addGroup("Existing"), lfs::core::NULL_NODE);
        auto restored =
            hydrate_scene_graph(chapter, live, ScenePayloadResolver{});
        ASSERT_FALSE(restored);
        EXPECT_EQ(restored.error().code(), lfs::ErrorCode::DataLoss);
        EXPECT_EQ(live.getNodeCount(), 1u);
        EXPECT_NE(live.getNode("Existing"), nullptr);
    }

    TEST(ProjectDocumentTest,
         InvalidPendingParametersRefuseHydrationBeforeSceneMutation) {
        auto document = ProjectDocument::create(fixed_uuid(910), 100);
        ASSERT_TRUE(document)
            << lfs::format_for_developer(document.error());
        require_status(document->edit_parameters().dom().set(
            "active_strategy", std::string{"not-a-strategy"}));

        Scene live;
        ASSERT_NE(live.addGroup("Existing"), lfs::core::NULL_NODE);
        auto restored = document->hydrate(live);
        ASSERT_FALSE(restored);
        EXPECT_EQ(restored.error().code(), lfs::ErrorCode::DataLoss);
        EXPECT_EQ(live.getNodeCount(), 1u);
        EXPECT_NE(live.getNode("Existing"), nullptr);
    }

    TEST(ProjectDocumentTest,
         PhaseAFailureInjectionLeavesEveryLiveSceneByteUntouched) {
        TemporaryDirectory temporary;
        const fs::path path =
            temporary.path / "phase-a-fixture.licht";
        write_phase_a_fixture(path);

        enum class Injection {
            Proj,
            Refs,
            Prms,
            Scng,
            Splt,
            Pcld,
            Mesh,
            SelmMismatch,
            MissingPayload,
        };
        const std::array injections{
            std::pair{Injection::Proj, "hostile PROJ"},
            std::pair{Injection::Refs, "hostile REFS"},
            std::pair{Injection::Prms, "hostile PRMS"},
            std::pair{Injection::Scng, "hostile SCNG"},
            std::pair{Injection::Splt, "hostile SPLT"},
            std::pair{Injection::Pcld, "hostile PCLD"},
            std::pair{Injection::Mesh, "hostile MESH"},
            std::pair{Injection::SelmMismatch,
                      "SELM topology mismatch"},
            std::pair{Injection::MissingPayload,
                      "missing embedded payload"},
        };
        const Uuid root_uuid = fixed_uuid(951);
        const Uuid splat_uuid = fixed_uuid(952);
        const Uuid point_uuid = fixed_uuid(953);
        const Uuid mesh_uuid = fixed_uuid(954);
        const Uuid reference_uuid = fixed_uuid(955);

        for (const auto& [injection, name] :
             injections) {
            SCOPED_TRACE(name);
            auto document = ProjectDocument::open(path);
            ASSERT_TRUE(document)
                << lfs::format_for_developer(
                       document.error());
            switch (injection) {
            case Injection::Proj:
                require_status(
                    document->edit_project().dom().set_json(
                        "georeference.world_origin",
                        lfs::io::JsonChapterDom::Json::
                            array({1.0, 2.0})));
                break;
            case Injection::Refs: {
                auto row =
                    document->edit_references()
                        .dom()
                        .array_find(
                            "references",
                            reference_uuid.to_string());
                ASSERT_TRUE(row);
                require_status(row->set(
                    "fingerprint.size",
                    std::string{"not-a-size"}));
                break;
            }
            case Injection::Prms:
                require_status(
                    document->edit_parameters().dom().set(
                        "active_strategy",
                        std::string{"hostile"}));
                break;
            case Injection::Scng: {
                auto row =
                    document->edit_scene_graph()
                        .dom()
                        .array_find(
                            "nodes",
                            root_uuid.to_string());
                ASSERT_TRUE(row);
                require_status(row->set(
                    "type",
                    std::string{"hostile-node"}));
                break;
            }
            case Injection::Splt: {
                auto* payload =
                    document->edit_splat(splat_uuid);
                ASSERT_NE(payload, nullptr);
                *payload = SplatChapterPayload{};
                break;
            }
            case Injection::Pcld: {
                auto* payload =
                    document->edit_point_cloud(point_uuid);
                ASSERT_NE(payload, nullptr);
                payload->point_cloud().reset();
                break;
            }
            case Injection::Mesh: {
                auto* payload =
                    document->edit_mesh(mesh_uuid);
                ASSERT_NE(payload, nullptr);
                payload->mesh().reset();
                break;
            }
            case Injection::SelmMismatch:
                require_status(
                    document->edit_selection().upsert_slice(
                        SelectionMaskSlice{
                            .node_uuid = point_uuid,
                            .domain =
                                lfs::core::
                                    SelectionDomain::
                                        PointCloud,
                            .mask = {4},
                        }));
                break;
            case Injection::MissingPayload:
                EXPECT_TRUE(
                    document->remove_point_cloud(point_uuid));
                break;
            }

            Scene live;
            const auto live_root =
                live.restoreNodeWithUuid(
                    Scene::RestoreNodeDesc{
                        .uuid = fixed_uuid(970),
                        .type = NodeType::GROUP,
                        .name = "Live sentinel",
                    });
            ASSERT_NE(live_root, lfs::core::NULL_NODE);
            const Uuid live_point_uuid = fixed_uuid(971);
            ASSERT_NE(
                live.restoreNodeWithUuid(
                    Scene::RestoreNodeDesc{
                        .uuid = live_point_uuid,
                        .type = NodeType::POINTCLOUD,
                        .name = "Live points",
                        .parent = live_root,
                        .point_cloud =
                            make_point_cloud(3),
                    }),
                lfs::core::NULL_NODE);
            const auto live_group =
                live.addSelectionGroup(
                    "Live selection",
                    {0.8f, 0.4f, 0.2f});
            live.setActiveSelectionGroup(live_group);
            live.applyPerNodeSelectionSlices(
                lfs::core::SelectionDomain::PointCloud,
                {{live_point_uuid,
                  selection_tensor(
                      {live_group, 0, live_group})}});
            live.updateSelectionGroupCounts();

            const auto before = witness_scene(live);
            auto staged =
                document->stage_hydration(live);
            ASSERT_FALSE(staged);
            EXPECT_TRUE(witness_scene(live) == before);
        }
    }

    TEST(ProjectDocumentTest,
         RepresentativeEditedSceneRoundTripsAndCleanRowsReuseSpans) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "representative.licht";

        const Uuid project_uuid = fixed_uuid(1);
        const Uuid root_uuid = fixed_uuid(2);
        const Uuid training_uuid = fixed_uuid(3);
        const Uuid imported_uuid = fixed_uuid(4);
        const Uuid point_uuid = fixed_uuid(5);
        const Uuid mesh_uuid = fixed_uuid(6);
        const Uuid checkpoint_uuid = fixed_uuid(7);

        Scene source;
        const auto root = source.restoreNodeWithUuid(Scene::RestoreNodeDesc{
            .uuid = root_uuid,
            .type = NodeType::GROUP,
            .name = "Assets",
        });
        ASSERT_NE(root, lfs::core::NULL_NODE);
        const auto training = source.restoreNodeWithUuid(
            Scene::RestoreNodeDesc{
                .uuid = training_uuid,
                .type = NodeType::SPLAT,
                .name = "Training model",
                .parent = root,
                .gaussian_count = 2,
                .model = make_splat(2),
            });
        ASSERT_NE(training, lfs::core::NULL_NODE);
        glm::mat4 edited_transform{1.0f};
        edited_transform[3] = glm::vec4(3.0f, 4.0f, 5.0f, 1.0f);
        const auto imported = source.restoreNodeWithUuid(
            Scene::RestoreNodeDesc{
                .uuid = imported_uuid,
                .type = NodeType::SPLAT,
                .name = "Edited import",
                .parent = root,
                .gaussian_count = 3,
                .local_transform = edited_transform,
                .visible = false,
                .locked = true,
                .payload_diverged = true,
                .model = make_splat(3),
            });
        ASSERT_NE(imported, lfs::core::NULL_NODE);
        auto point_cloud = make_point_cloud(2);
        const auto point = source.restoreNodeWithUuid(
            Scene::RestoreNodeDesc{
                .uuid = point_uuid,
                .type = NodeType::POINTCLOUD,
                .name = "Survey points",
                .parent = root,
                .georef_pose =
                    lfs::core::GeoreferencePose{
                        .translation = glm::dvec3(1.0, 2.0, 3.0),
                    },
                .point_cloud = point_cloud,
            });
        ASSERT_NE(point, lfs::core::NULL_NODE);
        auto mesh = make_mesh();
        const auto mesh_node = source.restoreNodeWithUuid(
            Scene::RestoreNodeDesc{
                .uuid = mesh_uuid,
                .type = NodeType::MESH,
                .name = "Reference mesh",
                .parent = root,
                .mesh = mesh,
            });
        ASSERT_NE(mesh_node, lfs::core::NULL_NODE);
        source.setTrainingModelNode(training_uuid);
        ASSERT_EQ(source.getTrainingModelNodeUuid(), training_uuid);

        const std::uint8_t group =
            source.addSelectionGroup("Review", {0.2f, 0.4f, 0.8f});
        source.setActiveSelectionGroup(group);
        source.applyPerNodeSelectionSlices(
            lfs::core::SelectionDomain::Splat,
            {{training_uuid, selection_tensor({0, 0})},
             {imported_uuid, selection_tensor({group, 0, group})}});
        source.applyPerNodeSelectionSlices(
            lfs::core::SelectionDomain::PointCloud,
            {{point_uuid, selection_tensor({0, group})}});

        ScenePayloadBindings bindings{
            {training_uuid,
             PayloadBinding{
                 .fourcc = "CKPT",
                 .instance_uuid = checkpoint_uuid,
                 .source_kind = "checkpoint",
             }},
            {imported_uuid,
             PayloadBinding{
                 .fourcc = "SPLT",
                 .instance_uuid = imported_uuid,
                 .source_kind = "ply",
             }},
            {point_uuid,
             PayloadBinding{
                 .fourcc = "PCLD",
                 .instance_uuid = point_uuid,
                 .source_kind = "ply",
             }},
            {mesh_uuid,
             PayloadBinding{
                 .fourcc = "MESH",
                 .instance_uuid = mesh_uuid,
                 .source_kind = "obj",
             }},
        };
        auto scene_chapter = capture_scene_graph(source, bindings);
        ASSERT_TRUE(scene_chapter)
            << lfs::format_for_developer(scene_chapter.error());
        const std::array selected_nodes{imported_uuid, point_uuid};
        auto selection_chapter =
            capture_selection_chapter(source, selected_nodes);
        ASSERT_TRUE(selection_chapter)
            << lfs::format_for_developer(selection_chapter.error());
        auto splat_payload = SplatChapterPayload::capture(
            *source.getNodeByUuid(imported_uuid)->model,
            SplatSourceKind::ImportedPly, false);
        ASSERT_TRUE(splat_payload)
            << lfs::format_for_developer(splat_payload.error());

        auto document = ProjectDocument::create(project_uuid, 100);
        ASSERT_TRUE(document)
            << lfs::format_for_developer(document.error());
        document->edit_scene_graph() = std::move(*scene_chapter);
        document->edit_selection() = std::move(*selection_chapter);
        require_status(document->set_splat(
            imported_uuid, std::move(*splat_payload)));
        require_status(document->set_point_cloud(
            point_uuid, PointCloudPayload(point_cloud)));
        require_status(
            document->set_mesh(mesh_uuid, MeshPayload(mesh)));

        auto& project = document->edit_project();
        for (const auto& [node, fourcc, reason, tag, locator] :
             std::array{
                 std::tuple{imported_uuid, std::string{"SPLT"},
                            std::string{"imported PLY"}, std::uint8_t{1},
                            std::string{"assets/source.ply"}},
                 std::tuple{point_uuid, std::string{"PCLD"},
                            std::string{"imported point cloud"},
                            std::uint8_t{2},
                            std::string{"assets/points.ply"}},
                 std::tuple{mesh_uuid, std::string{"MESH"},
                            std::string{"imported mesh"}, std::uint8_t{3},
                            std::string{"assets/mesh.obj"}},
             }) {
            require_status(project.upsert_embed_decision(EmbedDecision{
                .uuid = node,
                .node_uuid = node,
                .payload_fourcc = fourcc,
                .decision = "embedded",
                .reason = reason,
            }));
            require_status(project.upsert_embedded_payload_provenance(
                provenance(node, fourcc, locator, tag)));
        }
        require_status(project.set_georeference(ProjectGeoreference{
            .crs = "EPSG:2056",
            .world_origin = {2'600'000.25, 1'200'000.5, 450.0},
            .world_unit_scale = 1.0,
            .world_origin_provenance =
                WorldOriginProvenance::CentralizeByPointCloud,
        }));

        auto first = document->save(path, save_options(100, 200));
        ASSERT_TRUE(first) << lfs::format_for_developer(first.error());
        EXPECT_EQ(first->generation, 1u);
        EXPECT_EQ(first->rewritten_chunks, 8u);

        auto reopened = ProjectDocument::open(path);
        ASSERT_TRUE(reopened)
            << lfs::format_for_developer(reopened.error());
        EXPECT_EQ(reopened->splat_uuids(),
                  (std::vector<Uuid>{imported_uuid}));
        EXPECT_EQ(reopened->point_cloud_uuids(),
                  (std::vector<Uuid>{point_uuid}));
        EXPECT_EQ(reopened->mesh_uuids(),
                  (std::vector<Uuid>{mesh_uuid}));

        Scene restored;
        ASSERT_NE(
            restored.addGroup("Pre-commit sentinel"),
            lfs::core::NULL_NODE);
        const auto before_commit = witness_scene(restored);
        ScenePayloadResolver external{
            .splat =
                [&](const PayloadBinding& binding)
                -> lfs::Result<std::unique_ptr<lfs::core::SplatData>> {
                EXPECT_EQ(binding.fourcc, "CKPT");
                EXPECT_EQ(binding.instance_uuid, checkpoint_uuid);
                return make_splat(2);
            },
        };
        auto staged =
            reopened->stage_hydration(restored, external);
        ASSERT_TRUE(staged)
            << lfs::format_for_developer(staged.error());
        EXPECT_TRUE(witness_scene(restored) == before_commit);
        auto hydrated =
            ProjectDocument::commit_hydration(
                restored, std::move(*staged));
        EXPECT_EQ(restored.getNodeCount(), 5u);
        EXPECT_EQ(restored.getTrainingModelNodeUuid(), training_uuid);
        const auto* restored_import =
            restored.getNodeByUuid(imported_uuid);
        ASSERT_NE(restored_import, nullptr);
        EXPECT_EQ(restored_import->name, "Edited import");
        EXPECT_FALSE(restored_import->visible.get());
        EXPECT_TRUE(restored_import->locked.get());
        EXPECT_TRUE(restored_import->payload_diverged);
        EXPECT_FLOAT_EQ(restored_import->local_transform.get()[3].x,
                        3.0f);
        ASSERT_NE(restored.getNodeByUuid(point_uuid), nullptr);
        ASSERT_TRUE(restored.getNodeByUuid(point_uuid)->georef_pose);
        EXPECT_DOUBLE_EQ(
            restored.getNodeByUuid(point_uuid)->georef_pose->translation.y,
            2.0);
        EXPECT_EQ(hydrated.selection.selected_node_uuids,
                  (std::vector<Uuid>{imported_uuid, point_uuid}));
        EXPECT_EQ(
            restored.capturePerNodeSelectionSlices(
                        lfs::core::SelectionDomain::Splat)
                .at(imported_uuid)
                .cpu()
                .to_vector_uint8(),
            (std::vector<std::uint8_t>{group, 0, group}));
        EXPECT_EQ(
            restored.capturePerNodeSelectionSlices(
                        lfs::core::SelectionDomain::PointCloud)
                .at(point_uuid)
                .cpu()
                .to_vector_uint8(),
            (std::vector<std::uint8_t>{0, group}));

        auto second = reopened->save(path, save_options(110, 300));
        ASSERT_TRUE(second)
            << lfs::format_for_developer(second.error());
        EXPECT_EQ(second->generation, 2u);
        EXPECT_EQ(second->rewritten_chunks, 1u);
        EXPECT_EQ(second->reused_chunks, 7u);

        auto reader = ProjectReader::open(path);
        ASSERT_TRUE(reader)
            << lfs::format_for_developer(reader.error());
        EXPECT_EQ(reader->find(FOURCC_PROJ, project_uuid)->source_generation,
                  2u);
        EXPECT_EQ(reader->find(FOURCC_REFS, project_uuid)->source_generation,
                  1u);
        EXPECT_EQ(reader->find(FOURCC_SPLT, imported_uuid)
                          ->payload_offset %
                      TENSOR_PAYLOAD_ALIGNMENT,
                  0u);
    }

    TEST(ProjectDocumentTest,
         MissingReferencesRemainVerbatimAndReverseOwnersInvert) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "missing-references.licht";
        const Uuid project_uuid = fixed_uuid(200);
        const Uuid dataset_uuid = fixed_uuid(201);
        const Uuid rad_reference_uuid = fixed_uuid(202);
        const Uuid environment_uuid = fixed_uuid(203);
        const Uuid dataset_node_uuid = fixed_uuid(204);
        const Uuid rad_node_uuid = fixed_uuid(205);

        auto document = ProjectDocument::create(project_uuid, 1000);
        ASSERT_TRUE(document)
            << lfs::format_for_developer(document.error());
        auto& references = document->edit_references();
        for (const auto& [uuid, key, kind, locator, tag] :
             std::array{
                 std::tuple{dataset_uuid, std::string{"dataset.root"},
                            std::string{"dataset"},
                            std::string{"../absent-dataset"},
                            std::uint8_t{10}},
                 std::tuple{rad_reference_uuid,
                            std::string{"splat.live_rad"},
                            std::string{"rad"},
                            std::string{"assets/absent.rad"},
                            std::uint8_t{11}},
                 std::tuple{environment_uuid,
                            std::string{"view.environment"},
                            std::string{"environment_map"},
                            std::string{"assets/absent.hdr"},
                            std::uint8_t{12}},
             }) {
            require_status(references.upsert(ReferenceRecord{
                .uuid = uuid,
                .key = key,
                .kind = kind,
                .locator =
                    {.preferred = locator,
                     .base = LocatorBase::Project,
                     .absolute_fallback =
                         std::format("/stale/{}", locator)},
                .fingerprint = fake_fingerprint(tag),
                .unresolved = true,
            }));
        }
        require_status(
            document->edit_project().set_dataset_reference(dataset_uuid));
        require_status(document->edit_project().upsert_embed_decision(
            EmbedDecision{
                .uuid = rad_node_uuid,
                .node_uuid = rad_node_uuid,
                .payload_fourcc = "REFS",
                .decision = "external",
                .reference_uuid = rad_reference_uuid,
                .reason = "live RAD remains external",
            }));

        auto& scene = document->edit_scene_graph();
        require_status(scene.upsert_node(SceneNodeRecord{
            .uuid = dataset_node_uuid,
            .type = "dataset",
            .name = "Missing dataset",
            .child_order = 0,
        }));
        require_status(scene.upsert_node(SceneNodeRecord{
            .uuid = rad_node_uuid,
            .type = "splat",
            .name = "Missing live RAD",
            .parent_uuid = dataset_node_uuid,
            .child_order = 0,
            .payload =
                PayloadBinding{
                    .fourcc = "REFS",
                    .instance_uuid = rad_reference_uuid,
                    .reference_uuid = rad_reference_uuid,
                    .source_kind = "rad",
                },
        }));

        auto first = document->save(path, save_options(210, 1100));
        ASSERT_TRUE(first) << lfs::format_for_developer(first.error());
        auto first_reader = ProjectReader::open(path);
        ASSERT_TRUE(first_reader);
        const ChunkInfo first_refs =
            *first_reader->find(FOURCC_REFS, project_uuid);
        auto first_bytes = first_reader->read_chunk(first_refs);
        ASSERT_TRUE(first_bytes);

        auto reopened = ProjectDocument::open(path);
        ASSERT_TRUE(reopened)
            << lfs::format_for_developer(reopened.error());
        auto records = reopened->references().records();
        ASSERT_TRUE(records);
        ASSERT_EQ(records->size(), 3u);
        EXPECT_TRUE(std::ranges::all_of(
            *records,
            [](const ReferenceRecord& record) {
                return record.unresolved;
            }));
        auto rad_node = reopened->scene_graph().find(rad_node_uuid);
        ASSERT_TRUE(rad_node);
        ASSERT_TRUE(*rad_node);
        ASSERT_TRUE((*rad_node)->payload);
        EXPECT_EQ((*rad_node)->payload->reference_uuid,
                  rad_reference_uuid);

        const std::array additional{
            ReferenceOwnerBinding{
                .reference_uuid = environment_uuid,
                .chapter = "VIEW",
                .owner_uuid = std::nullopt,
                .field = "environment_reference",
            },
        };
        auto reverse =
            reopened->reverse_reference_index(additional);
        ASSERT_TRUE(reverse)
            << lfs::format_for_developer(reverse.error());
        EXPECT_EQ(reverse->at(dataset_uuid).front().chapter, "PROJ");
        EXPECT_EQ(reverse->at(rad_reference_uuid).front().chapter, "SCNG");
        EXPECT_EQ(reverse->at(rad_reference_uuid).front().owner_uuid,
                  rad_node_uuid);
        EXPECT_EQ(reverse->at(environment_uuid).front().chapter, "VIEW");

        auto second = reopened->save(path, save_options(220, 1200));
        ASSERT_TRUE(second)
            << lfs::format_for_developer(second.error());
        EXPECT_EQ(second->rewritten_chunks, 1u);
        auto second_reader = ProjectReader::open(path);
        ASSERT_TRUE(second_reader);
        const ChunkInfo* second_refs =
            second_reader->find(FOURCC_REFS, project_uuid);
        ASSERT_NE(second_refs, nullptr);
        EXPECT_EQ(second_refs->source_generation, 1u);
        EXPECT_EQ(second_refs->header_offset, first_refs.header_offset);
        auto second_bytes = second_reader->read_chunk(*second_refs);
        ASSERT_TRUE(second_bytes);
        EXPECT_EQ(*second_bytes, *first_bytes);

        auto relink_document = ProjectDocument::open(path);
        ASSERT_TRUE(relink_document);
        const auto before =
            relink_document->references().find(dataset_uuid);
        ASSERT_TRUE(before);
        auto relink = relink_document->edit_references().relink(
            dataset_uuid,
            ReferenceLocator{
                .preferred = "../still-absent",
                .base = LocatorBase::Project,
            },
            temporary.path / "does-not-exist");
        EXPECT_FALSE(relink);
        const auto after =
            relink_document->references().find(dataset_uuid);
        ASSERT_TRUE(after);
        EXPECT_EQ(*after, *before);
    }

} // namespace
