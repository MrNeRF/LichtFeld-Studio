/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "collision_fixture.hpp"
#include "core/collision_serialization.hpp"
#include "core/splat_data.hpp"
#include "io/project_document.hpp"
#include "licht_test_support.hpp"
#include "training/project_snapshot_chapters.hpp"
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <unordered_map>

namespace {
    using namespace lfs::core;
    using namespace lfs::io::project;
    using namespace lfs::test::licht;
    using collision_test::box;

    class CollisionArchive : public ::testing::Test {
    protected:
        TemporaryDirectory temporary;
        const std::filesystem::path path = temporary.path / "collision.licht";
        Scene scene;
        std::shared_ptr<CollisionVolume> volume = std::make_shared<CollisionVolume>(*box({1, 2, 3}, {3, 1, 2}, .0125f));
        void SetUp() override {
            volume->settings.opacity = .35f;
            volume->settings.max_gaussian_area = 2.5f;
            volume->settings.auto_resolution = true;
            volume->settings.cleanup = true;
            const auto parent = scene.addGroup("Parent");
            scene.setNodeTransform(parent, glm::translate(glm::mat4(1), {4, 5, 6}));
            for (int i = 0; i < 13; ++i) {
                const auto id = scene.addCollision(std::to_string(i), i < 12 ? volume : box({0, 0, 0}, {1, 1, 1}), parent);
                scene.setNodeTransform(id, glm::translate(glm::mat4(1), {float(i), 2, 3}));
                scene.getNodeById(id)->visible = i != 0;
            }
            auto document = make_empty_document(generate_uuid_v4());
            auto captured = require_result(capture_scene_graph_state(scene, {}));
            for (const auto& [uuid, payload] : captured.collisions)
                require_status(document->set_collision(uuid, payload));
            document->edit_scene_graph() = require_result(materialize_scene_graph_chapter(std::move(captured)));
            (void)require_result(document->save(path));
        }
    };

    TEST_F(CollisionArchive, BinaryDedupDeferredHydrationAndSaveAsPreserveNodes) {
        auto reader = require_result(ProjectReader::open(path));
        EXPECT_EQ(std::ranges::count_if(reader.chunks(), [](const auto& row) {
                      return row.is_live() && row.key.fourcc == FOURCC_COLV;
                  }),
                  2);
        EXPECT_TRUE(reader.commit().required_reader_capabilities.contains(COLLISION_VOLUMES_V1));
        EXPECT_TRUE(reader.commit().required_writer_capabilities.contains(COLLISION_VOLUMES_V1));
        // Assert wire positions independently of the enum to catch renumbering.
        EXPECT_TRUE(reader.commit().required_reader_capabilities.contains(11));
        EXPECT_TRUE(reader.commit().required_writer_capabilities.contains(11));
        EXPECT_FALSE(reader.commit().required_reader_capabilities.contains(10));
        EXPECT_FALSE(reader.commit().required_writer_capabilities.contains(10));
        ReaderOptions old_reader;
        old_reader.reader_capabilities.set(11, false);
        auto incompatible = ProjectReader::open(path, old_reader);
        ASSERT_FALSE(incompatible);
        EXPECT_EQ(incompatible.error().code(), lfs::ErrorCode::Unsupported);
        EXPECT_NE(incompatible.error().user_message().find("newer LichtFeld"), std::string_view::npos);
        auto document = require_result(ProjectDocument::open(path, {.defer_geometry_payloads = true}));
        EXPECT_LT(document.scene_graph().to_bytes().size(), 16 * 1024);
        for (const auto& state : document.payload_states())
            EXPECT_FALSE(state.loaded);
        const auto unloaded_copy = temporary.path / "unloaded.licht";
        ASSERT_TRUE(document.save_as(unloaded_copy));
        for (const auto& state : document.payload_states())
            EXPECT_FALSE(state.loaded);
        Scene target;
        ASSERT_TRUE(document.hydrate(target));
        auto reopened = require_result(ProjectDocument::open(unloaded_copy, {.defer_geometry_payloads = true}));
        auto shell = require_result(reopened.stage_shell(target));
        const auto payloads = require_result(capture_scene_graph_state(scene, {})).collisions;
        std::unordered_map<Uuid, int> resolutions;
        ScenePayloadResolver resolver;
        resolver.collision = [&](const Uuid& uuid) -> lfs::Result<std::shared_ptr<const CollisionVolume>> {
            ++resolutions[uuid];
            return payloads.at(uuid);
        };
        target.addGroup("Replaced by direct adapter hydration");
        ASSERT_TRUE(hydrate_scene_graph(reopened.scene_graph(), target, resolver));
        ASSERT_EQ(resolutions.size(), 2);
        for (const auto& [uuid, count] : resolutions)
            EXPECT_EQ(count, 1); // Adapter caching is independent of the document's cache.
        const auto copy = temporary.path / "loaded.licht";
        ASSERT_TRUE(reopened.save_as(copy));
        for (const auto& saved_path : {unloaded_copy, copy}) {
            auto saved = require_result(ProjectReader::open(saved_path));
            EXPECT_TRUE(saved.commit().required_reader_capabilities.contains(11));
            EXPECT_TRUE(saved.commit().required_writer_capabilities.contains(11));
            EXPECT_FALSE(saved.commit().required_reader_capabilities.contains(10));
            EXPECT_FALSE(saved.commit().required_writer_capabilities.contains(10));
        }
        auto copied = require_result(ProjectDocument::open(copy));
        Scene loaded;
        ASSERT_TRUE(copied.hydrate(loaded));
        const auto expected = require_result(capture_scene_graph(scene, {})).to_bytes();
        for (const auto* restored : {shell.get(), &target, &loaded}) {
            // Recapture checks local/parent transforms, visibility, node type and shared payload references.
            EXPECT_EQ(require_result(capture_scene_graph(*restored, {})).to_bytes(), expected);
            for (const auto* node : restored->getNodes()) {
                if (!node->collision)
                    continue;
                EXPECT_FALSE(node->mesh);
                EXPECT_EQ(node->gaussian_count.load(), 0);
                EXPECT_EQ(encodeCollision(*node->collision), encodeCollision(*scene.getNodeByUuid(node->uuid)->collision));
            }
        }
    }

    TEST_F(CollisionArchive, ReadAndSaveFailuresAreTransactional) {
        auto document = require_result(ProjectDocument::open(path));
        const auto baseline = read_file_bytes(path);
        const auto bytes = encodeCollision(*volume);
        EXPECT_EQ(bytes.size(), collision_payload_header_bytes + volume->words.size() * sizeof(uint32_t));
        EXPECT_THROW(decodeCollision(std::span(bytes).first(bytes.size() - 1)), std::invalid_argument);
        EXPECT_THROW(decodeCollision(bytes, bytes.size() - 1), std::length_error);
        EXPECT_THROW(encodeCollision(*volume, bytes.size() - 1), std::length_error);
        auto budget = ProjectDocument::open(path, {.collision_byte_budget = 1});
        ASSERT_FALSE(budget);
        EXPECT_EQ(budget.error().code(), lfs::ErrorCode::ResourceExhausted);
        auto invalid = std::make_shared<CollisionVolume>(*volume);
        invalid->words.front() = 0xffffffff;
        EXPECT_FALSE(document.set_collision(document.payload_states().front().instance_uuid, invalid));
        std::filesystem::create_directory(temporary.path / "directory");
        EXPECT_FALSE(document.save(temporary.path / "directory"));
        EXPECT_EQ(read_file_bytes(path), baseline);
        ASSERT_TRUE(document.save(path));
        for (const bool erase : {false, true}) {
            auto reader = require_result(ProjectReader::open(path));
            auto writer = require_result(ProjectWriter::append(path));
            require_status(writer.plan_commit());
            auto corrupt = encodeCollision(*volume);
            corrupt[0] = std::byte{0};
            require_status(writer.preflight(corrupt.size()));
            for (const auto& row : reader.chunks()) {
                if (!row.is_live())
                    continue;
                if (row.key.fourcc == FOURCC_COLV)
                    require_status(erase ? writer.erase(row.key) : writer.write_chunk(row.key, corrupt));
                else
                    require_status(writer.reuse_if_clean(require_result(reader.make_clean_proof(row, 1)), 1));
            }
            require_status(writer.commit());
            auto failed = ProjectDocument::open(path);
            ASSERT_FALSE(failed);
            EXPECT_EQ(failed.error().code(), lfs::ErrorCode::DataLoss);
        }
        Scene unchanged;
        const auto retained = unchanged.addGroup("Retained");
        EXPECT_FALSE(hydrate_scene_graph(require_result(capture_scene_graph(scene, {})), unchanged, {}));
        EXPECT_EQ(unchanged.getNodes().size(), 1);
        EXPECT_NE(unchanged.getNodeById(retained), nullptr);
    }

    TEST(CollisionScene, CpuSourceAndSafePointSnapshotOwnDataAfterDeletion) {
        Scene scene;
        const auto parent = scene.addGroup("Parent");
        scene.setNodeTransform(parent, glm::scale(glm::mat4(1), glm::vec3(10)));
        auto model = make_splat(1);
        model->means() = Tensor::from_vector({1.f, 2.f, 3.f}, {1, 3}, Device::CPU);
        model->deleted() = Tensor::ones_bool({1}, Device::CPU);
        const auto source = scene.addSplat("Selected hidden source", std::move(model), parent);
        scene.addSplat("Unselected", make_splat(1));
        scene.setTrainingModelNode(source);
        scene.setNodeVisibility(source, false);
        const auto volume = box({1, 2, 3}, {3, 1, 2});
        const auto first = scene.addCollision("First", volume, source);
        const auto second = scene.addCollision("Second", volume, source);
        const auto transform = glm::translate(glm::mat4(1), {-4, 5, -6});
        scene.setNodeTransform(source, transform);
        scene.setNodeTransform(first, transform);
        for (const auto invalid : {NULL_NODE, first})
            EXPECT_THROW((void)scene.captureCollisionSources(invalid), std::runtime_error);
        const auto sources = scene.captureCollisionSources(source);
        ASSERT_EQ(sources.size(), 1);
        EXPECT_EQ(sources.front().transform, scene.getWorldTransform(source));
        EXPECT_TRUE(sources.front().deleted.item<bool>());
        scene.removeNode("Unselected"); // Only the training source has a checkpoint binding.
        lfs::training::ProjectSnapshotCpuState captured;
        ASSERT_TRUE(lfs::training::capture_project_snapshot_cpu_state(scene, ParameterManagerSnapshot{}, generate_uuid_v4(), 3, captured));
        const auto first_uuid = scene.getNodeUuid(first), second_uuid = scene.getNodeUuid(second);
        scene.removeNodeById(source);
        EXPECT_EQ(sources.front().means.slice(1, 0, 1).item<float>(), 1);
        EXPECT_THROW((void)scene.captureCollisionSources(source), std::runtime_error);
        lfs::training::ProjectSnapshotChapters chapters;
        ASSERT_TRUE(lfs::training::materialize_project_snapshot_cpu_chapters(std::move(captured), chapters));
        ASSERT_EQ(chapters.collisions.size(), 1);
        EXPECT_EQ(chapters.collisions.begin()->second, volume);
        const auto one = chapters.scene_graph.find(first_uuid), two = chapters.scene_graph.find(second_uuid);
        ASSERT_TRUE(one && *one && two && *two);
        EXPECT_EQ((**one).collision, (**two).collision);
        EXPECT_TRUE((**one).parent_uuid);
        for (int axis = 0; axis < 3; ++axis)
            EXPECT_EQ((**one).local_transform[12 + axis], transform[3][axis]);
    }
} // namespace
