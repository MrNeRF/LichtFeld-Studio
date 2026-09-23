/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/checkpoint_format.hpp"
#include "io/project_chapters.hpp"
#include "io/project_container.hpp"
#include "io/project_inspector.hpp"
#include "io/project_path.hpp"
#include "licht_test_support.hpp"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <span>
#include <thread>

namespace {

    namespace fs = std::filesystem;
    using namespace lfs::io::project;
    using namespace lfs::test::licht;

    CreateOptions create_options(const std::uint64_t tag) {
        return {
            .project_uuid = fixed_uuid(1000),
            .file_uuid = fixed_uuid(tag),
            .role = ContainerRole::Master,
            .creation_time_unix_ns = 1'700'000'000'000'000'000,
            .index_compression = IndexCompression::StoredForDeterministicTests,
            .disk_reserve_bytes = 0,
        };
    }

    CommitOptions commit_options(const std::uint64_t tag,
                                 const std::uint64_t generation,
                                 const Version minimum_reader = {1, 0}) {
        return {
            .kind = CommitKind::Explicit,
            .commit_uuid = fixed_uuid(tag),
            .snapshot_uuid = fixed_uuid(tag + 100),
            .wallclock_unix_ns = 1'700'000'000'000'000'000 + generation,
            .min_reader_version = minimum_reader,
        };
    }

    void write_generation(ProjectWriter& writer,
                          const CommitOptions& commit,
                          const ChunkKey& key,
                          const std::vector<std::byte>& payload,
                          const ChunkWriteOptions& chunk_options = {}) {
        require_status(writer.plan_commit(commit));
        require_status(writer.preflight(payload.size()));
        require_status(writer.write_chunk(key, payload, chunk_options));
        require_status(writer.commit());
    }

    fs::path make_project(const fs::path& path) {
        ProjectWriter writer = require_result(
            ProjectWriter::create(path, create_options(1001)));
        write_generation(writer, commit_options(1101, 1), fixed_key("PROJ", 1201),
                         byte_vector(R"({"name":"inspect"})"));
        return path;
    }

    void flip_byte(const fs::path& path, const std::uint64_t offset) {
        std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(stream);
        stream.seekg(static_cast<std::streamoff>(offset));
        char byte = 0;
        stream.read(&byte, 1);
        ASSERT_EQ(stream.gcount(), 1);
        stream.seekp(static_cast<std::streamoff>(offset));
        byte ^= static_cast<char>(0x5a);
        stream.write(&byte, 1);
    }

    std::vector<std::byte> checkpoint_payload(const std::int32_t iteration) {
        lfs::core::CheckpointHeader header;
        header.iteration = iteration;
        header.num_gaussians = 42;
        header.sh_degree = 2;
        std::vector<std::byte> payload(sizeof(header));
        std::memcpy(payload.data(), &header, sizeof(header));
        return payload;
    }

    TEST(ProjectInspector, CardClassifiesHeadsAndCommitCompatibility) {
        TemporaryDirectory temporary;
        const fs::path healthy = make_project(temporary.path / "healthy.licht");
        auto card = require_result(inspect_project_card(healthy));
        EXPECT_EQ(card.open_state, OpenState::Open);
        EXPECT_EQ(card.validation_scope, "head");
        EXPECT_EQ(card.generation, 1u);
        EXPECT_EQ(card.project_uuid, fixed_uuid(1000));

        const fs::path one_bad = temporary.path / "one-bad.licht";
        fs::copy_file(healthy, one_bad);
        const auto healthy_reader = require_result(ProjectReader::open(healthy));
        const auto bad_slot = 1u - healthy_reader.selected_head().slot_id;
        flip_byte(one_bad, HEAD_SLOT_OFFSETS[bad_slot] + 200);
        card = require_result(inspect_project_card(one_bad));
        EXPECT_EQ(card.open_state, OpenState::Open);
        EXPECT_NE(card.diagnostic.find("head slot"), std::string::npos);

        const fs::path both_bad = temporary.path / "both-bad.licht";
        fs::copy_file(healthy, both_bad);
        flip_byte(both_bad, HEAD_SLOT_OFFSETS[0] + 200);
        flip_byte(both_bad, HEAD_SLOT_OFFSETS[1] + 200);
        card = require_result(inspect_project_card(both_bad));
        EXPECT_EQ(card.open_state, OpenState::RepairOnly);

        const fs::path truncated = temporary.path / "truncated.licht";
        fs::copy_file(healthy, truncated);
        fs::resize_file(truncated, fs::file_size(truncated) - 1);
        card = require_result(inspect_project_card(truncated));
        EXPECT_EQ(card.open_state, OpenState::RepairOnly);

        const fs::path newer = temporary.path / "newer.licht";
        ProjectWriter newer_writer = require_result(
            ProjectWriter::create(newer, create_options(1002)));
        write_generation(newer_writer, commit_options(1102, 1, {99, 0}),
                         fixed_key("PROJ", 1202), byte_vector("newer"));
        card = require_result(inspect_project_card(newer));
        EXPECT_EQ(card.open_state, OpenState::UnsupportedNewer);
    }

    TEST(ProjectInspector, DetailsReportsHistoryStorageAndBudgetedCheckpoint) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "details.licht";
        const ChunkWriteOptions checkpoint_options{
            .chunk_version = 1,
            .compression = Compression::Stored,
            .tensor_payload = true,
        };
        const ChunkKey checkpoint = fixed_key("CKPT", 1301);
        {
            ProjectWriter writer = require_result(
                ProjectWriter::create(path, create_options(1003)));
            write_generation(writer, commit_options(1103, 1), checkpoint,
                             checkpoint_payload(7), checkpoint_options);
        }
        {
            ProjectWriter writer = require_result(ProjectWriter::append(
                path, AppendOptions{
                          .index_compression = IndexCompression::StoredForDeterministicTests,
                          .disk_reserve_bytes = 0,
                      }));
            require_status(writer.plan_commit(commit_options(1104, 2)));
            require_status(writer.preflight(0));
            require_status(writer.erase(checkpoint));
            require_status(writer.commit());
        }

        const fs::path sidecar = autosave_sidecar_path(path);
        std::ofstream(sidecar).put('\n');
        auto details = require_result(inspect_project_details(path));
        EXPECT_EQ(details.card.validation_scope, "structure");
        EXPECT_EQ(details.save_history.size(), 2u);
        EXPECT_GT(details.storage.dead_bytes, 0u);
        ASSERT_EQ(details.retained_checkpoints.size(), 1u);
        EXPECT_FALSE(details.retained_checkpoints.front().retained);
        EXPECT_EQ(details.retained_checkpoints.front().iteration, 7);
        EXPECT_TRUE(details.autosave_sidecar_present);

        details = require_result(inspect_project_details(
            path, sizeof(lfs::core::CheckpointHeader) - 1));
        ASSERT_EQ(details.retained_checkpoints.size(), 1u);
        EXPECT_FALSE(details.retained_checkpoints.front().header_reachable);
    }

    TEST(ProjectInspector, ConcurrentCardsDoNotHoldTheGILOrSharedState) {
        TemporaryDirectory temporary;
        const fs::path path = make_project(temporary.path / "parallel.licht");
        std::array<OpenState, 2> states{};
        std::array<std::thread, 2> workers;
        for (std::size_t index = 0; index < workers.size(); ++index) {
            workers[index] = std::thread([&, index] {
                states[index] = require_result(inspect_project_card(path)).open_state;
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        EXPECT_EQ(states[0], OpenState::Open);
        EXPECT_EQ(states[1], OpenState::Open);
    }

    TEST(ProjectInspector, FilterFactsMatchRealProjectContents) {
        TemporaryDirectory temporary;
        const auto write = [&](const std::string& name,
                               const std::vector<std::pair<ChunkKey, std::vector<std::byte>>>& chunks) {
            const auto path = temporary.path / (name + ".licht");
            auto writer = require_result(ProjectWriter::create(path, create_options(1401)));
            ProjectChapter project;
            require_status(project.set_project_uuid(fixed_uuid(1000)));
            require_status(project.set_created_at_unix_ns(1'700'000'000'000'000'000));
            const auto project_bytes = project.to_bytes();
            require_status(writer.plan_commit(commit_options(1401, 1)));
            std::uint64_t bytes = project_bytes.size();
            for (const auto& [key, payload] : chunks)
                bytes += payload.size();
            require_status(writer.preflight(bytes));
            require_status(writer.write_chunk(fixed_key("PROJ", 1000), project_bytes));
            for (const auto& [key, payload] : chunks)
                require_status(writer.write_chunk(key, payload));
            require_status(writer.commit());
            return require_result(ProjectReader::open(path));
        };

        SceneGraphChapter training_scene;
        const auto model_uuid = fixed_uuid(1402);
        require_status(training_scene.upsert_node(SceneNodeRecord{
            .uuid = model_uuid,
            .type = "splat",
            .name = "model",
            .payload = PayloadBinding{
                .fourcc = "CKPT",
                .instance_uuid = fixed_uuid(1403),
                .source_kind = "checkpoint",
            },
        }));
        require_status(training_scene.set_training_model_uuid(model_uuid));
        const auto checkpoint = checkpoint_payload(7);
        const auto training = write("project-a", {{fixed_key("SCNG", 1000), training_scene.to_bytes()},
                                                  {fixed_key("CKPT", 1403), checkpoint}});
        EXPECT_TRUE(inspect_project_filter_facts(training).has_checkpoint);
        EXPECT_FALSE(inspect_project_filter_facts(training).has_dataset);

        SceneGraphChapter no_training_scene;
        const auto unbound = write("project-b", {{fixed_key("SCNG", 1000), no_training_scene.to_bytes()},
                                                 {fixed_key("CKPT", 1403), checkpoint}});
        EXPECT_FALSE(inspect_project_filter_facts(unbound).has_checkpoint);
        EXPECT_FALSE(inspect_project_filter_facts(unbound).has_dataset);

        SceneGraphChapter dataset_scene;
        require_status(dataset_scene.upsert_node(SceneNodeRecord{
            .uuid = fixed_uuid(1404),
            .type = "dataset",
            .name = "project-data",
        }));
        const auto node_dataset = write("project-c", {{fixed_key("SCNG", 1000), dataset_scene.to_bytes()}});
        EXPECT_TRUE(inspect_project_filter_facts(node_dataset).has_dataset);

        ReferencesChapter references;
        require_status(references.upsert(ReferenceRecord{
            .uuid = fixed_uuid(1405),
            .key = "data.root",
            .kind = "images",
            .locator = {.preferred = "project-data", .base = LocatorBase::Project},
        }));
        const auto referenced_dataset = write("project-d", {{fixed_key("REFS", 1000), references.to_bytes()}});
        EXPECT_TRUE(inspect_project_filter_facts(referenced_dataset).has_dataset);

        ParameterManagerSnapshot snapshot;
        snapshot.mcmc_session = lfs::core::param::OptimizationParameters::mcmc_defaults();
        snapshot.mrnf_session = lfs::core::param::OptimizationParameters::mrnf_defaults();
        snapshot.igs_session = lfs::core::param::OptimizationParameters::igs_plus_defaults();
        snapshot.mcmc_current = snapshot.mcmc_session;
        snapshot.mrnf_current = snapshot.mrnf_session;
        snapshot.igs_current = snapshot.igs_session;

        ParametersChapter embedded_parameters;
        require_status(embedded_parameters.set_snapshot(snapshot));
        const auto embedded_bytes = byte_vector("{}");
        const auto embedded_uuid = fixed_uuid(1406);
        require_status(embedded_parameters.set_embedded_dataset(EmbeddedDatasetManifest{
            .schema_version = 1,
            .images_folder = "images",
            .complete = true,
            .entries = {{
                .rel_path = "metadata.json",
                .kind = "meta",
                .chunk_uuid = embedded_uuid,
                .bytes = embedded_bytes.size(),
                .xxh3_128 = xxh3_128(embedded_bytes),
            }},
        }));
        const auto embedded_dataset = write("project-e", {{fixed_key("PRMS", 1000), embedded_parameters.to_bytes()},
                                                          {fixed_key("DSRC", 1406), embedded_bytes}});
        EXPECT_TRUE(inspect_project_filter_facts(embedded_dataset).has_dataset);

        const auto plain = write("project-f", {});
        EXPECT_FALSE(inspect_project_filter_facts(plain).has_checkpoint);
        EXPECT_FALSE(inspect_project_filter_facts(plain).has_dataset);
    }

    TEST(ProjectInspector, ContentStampTracksSavedSplatAndViewAcrossIndexEncodings) {
        TemporaryDirectory temporary;
        const auto splat = fixed_key("SPLT", 1202);
        const auto view = fixed_key("VIEW", 1203);
        const auto initial_splat = byte_vector("splat-a");
        const auto initial_view = byte_vector("view-a");
        const auto create = [&](const fs::path& path,
                                const IndexCompression compression) {
            auto options = create_options(1001);
            options.index_compression = compression;
            ProjectWriter writer =
                require_result(ProjectWriter::create(path, options));
            require_status(writer.plan_commit(commit_options(1101, 1)));
            require_status(writer.preflight(32));
            require_status(writer.write_chunk(fixed_key("SCNG", 1201),
                                              byte_vector("graph")));
            require_status(writer.write_chunk(splat, initial_splat));
            require_status(writer.write_chunk(view, initial_view));
            require_status(writer.commit());
        };

        const auto stored = temporary.path / "project-a.licht";
        const auto compressed = temporary.path / "project-b.licht";
        create(stored, IndexCompression::StoredForDeterministicTests);
        create(compressed, IndexCompression::Zstd);
        EXPECT_EQ(require_result(ProjectReader::open(stored)).commit().index_compression,
                  Compression::Stored);
        EXPECT_EQ(require_result(ProjectReader::open(compressed)).commit().index_compression,
                  Compression::ZstdFramed);
        const auto original = project_content_stamp(compressed);
        ASSERT_FALSE(original.empty());
        EXPECT_EQ(project_content_stamp(stored), original);

        const auto append = [&](const std::uint64_t tag,
                                const std::uint64_t generation,
                                const ChunkKey* changed,
                                const std::vector<std::byte>& payload) {
            ProjectReader prior = require_result(ProjectReader::open(compressed));
            ProjectWriter writer = require_result(ProjectWriter::append(compressed));
            require_status(writer.plan_commit(commit_options(tag, generation)));
            require_status(writer.preflight(payload.size()));
            for (const auto& row : prior.chunks()) {
                if (row.is_live() && (!changed || row.key != *changed)) {
                    const auto proof =
                        require_result(prior.make_clean_proof(row, generation));
                    require_status(writer.reuse_if_clean(proof, generation));
                }
            }
            if (changed) {
                require_status(writer.write_chunk(*changed, payload));
            }
            require_status(writer.commit());
        };
        append(1102, 2, nullptr, {});
        EXPECT_EQ(project_content_stamp(compressed), original);

        append(1103, 3, &splat, byte_vector("splat-b"));
        const auto splat_changed = project_content_stamp(compressed);
        ASSERT_FALSE(splat_changed.empty());
        EXPECT_NE(splat_changed.substr(0, 64), original.substr(0, 64));
        EXPECT_EQ(splat_changed.substr(65), original.substr(65));

        append(1104, 4, &view, byte_vector("view-b"));
        const auto view_changed = project_content_stamp(compressed);
        ASSERT_FALSE(view_changed.empty());
        EXPECT_EQ(view_changed.substr(0, 64), splat_changed.substr(0, 64));
        EXPECT_NE(view_changed.substr(65), splat_changed.substr(65));

        const auto checkpoint = fixed_key("CKPT", 1204);
        append(1105, 5, &checkpoint, byte_vector("checkpoint"));
        const auto checkpoint_changed = project_content_stamp(compressed);
        EXPECT_NE(checkpoint_changed.substr(0, 64), view_changed.substr(0, 64));
        EXPECT_EQ(checkpoint_changed.substr(65), view_changed.substr(65));

        const auto no_view = temporary.path / "project-c.licht";
        ProjectWriter writer = require_result(
            ProjectWriter::create(no_view, create_options(1001)));
        write_generation(writer, commit_options(1101, 1),
                         fixed_key("SCNG", 1201), byte_vector("graph"));
        EXPECT_EQ(project_content_stamp(no_view).size(), 129u);
    }

} // namespace
