/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "app/headless_recovery_document.hpp"
#include "io/loaders/loader_utils.hpp"
#include "io/project/project_container_internal.hpp"
#include "io/project/span_streambuf.hpp"
#include "io/project_document.hpp"
#include "io/project_recovery.hpp"
#include "licht_test_support.hpp"
#include "project/session_state.hpp"
#include "training/checkpoint.hpp"
#include "training/project_snapshot_chapters.hpp"
#include "training/strategies/mcmc.hpp"
#include "visualizer/core/parameter_manager.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#if defined(__linux__)
#include <csignal>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

    namespace fs = std::filesystem;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::NodeType;
    using lfs::core::Scene;
    using lfs::core::Tensor;
    using lfs::core::Uuid;
    using namespace lfs::io::project;
    using lfs::test::licht::json_root;
    using lfs::test::licht::make_empty_document;
    using lfs::test::licht::make_point_cloud;
    using lfs::test::licht::make_splat;
    using lfs::test::licht::make_triangle_mesh;
    using lfs::test::licht::one_pixel_png;
    using lfs::test::licht::read_file_bytes;
    using lfs::test::licht::require_result;
    using lfs::test::licht::require_result_ptr;
    using lfs::test::licht::require_status;
    using lfs::test::licht::TemporaryDirectory;

    Uuid fixed_uuid(const std::uint64_t tag) {
        return lfs::test::licht::fixed_uuid_in_namespace(0x70000000, tag);
    }

    Tensor selection_tensor(
        const std::initializer_list<std::uint8_t> values) {
        auto result = Tensor::empty({values.size()}, Device::CPU,
                                    DataType::UInt8);
        std::ranges::copy(values, result.ptr<std::uint8_t>());
        return result;
    }

    ReferenceFingerprint fake_fingerprint(const std::uint8_t tag) {
        return lfs::test::licht::fingerprint(tag);
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

    ProjectDocumentSaveOptions save_options(
        const std::uint64_t identity_tag, const std::uint64_t wallclock) {
        return lfs::test::licht::deterministic_document_save_options(
            0x70000000, identity_tag, wallclock);
    }

    void inject_legacy_gui_window(
        const fs::path& path,
        const std::uint64_t identity_tag,
        const std::uint64_t wallclock) {
        auto reader = require_result(ProjectReader::open(path));
        const auto* gui_row = reader.find(
            FOURCC_GUIL, reader.superblock().project_uuid);
        if (!gui_row) {
            throw std::runtime_error("missing GUIL chapter");
        }
        auto gui_bytes = require_result(reader.read_chunk(*gui_row));
        auto gui_dom = require_result(lfs::io::JsonChapterDom::from_bytes(gui_bytes));
        auto gui = json_root(gui_dom);
        gui["layouts"][0]["areas"][0]["spaces"][0]
           ["opaque_payload"]["window"] = {
               {"x", 120},
               {"y", 80},
               {"width", 1920},
               {"height", 1080},
               {"maximized", false},
               {"fullscreen", false},
           };
        auto legacy_dom = require_result(lfs::io::JsonChapterDom::parse(gui.dump(2)));
        const auto legacy_bytes = legacy_dom.to_bytes();

        auto writer = require_result(ProjectWriter::append(
            path,
            AppendOptions{
                .compatibility = {},
                .index_compression =
                    IndexCompression::StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
                .boundary_observer = {},
            }));
        require_status(writer.plan_commit(CommitOptions{
            .kind = CommitKind::Explicit,
            .commit_uuid = fixed_uuid(identity_tag),
            .snapshot_uuid = reader.commit().snapshot_uuid,
            .wallclock_unix_ns = wallclock,
        }));
        require_status(writer.preflight(legacy_bytes.size()));
        for (const auto& row : reader.chunks()) {
            if (!row.is_live()) {
                continue;
            }
            if (row.key == gui_row->key) {
                require_status(writer.write_chunk(
                    row.key, legacy_bytes,
                    ChunkWriteOptions{
                        .chunk_version = row.chunk_version,
                        .compression = Compression::Stored,
                    }));
                continue;
            }
            auto proof = require_result(
                reader.make_clean_proof(row, 1));
            require_status(writer.reuse_if_clean(proof, 1));
        }
        require_status(writer.commit());
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

        auto document = make_empty_document(project_uuid, 100);
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
            mesh_uuid, MeshPayload(make_triangle_mesh())));
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

    void write_one_splat_document(
        const fs::path& path, const lfs::core::SplatData& model,
        const Uuid& project_uuid, const Uuid& splat_uuid,
        const std::uint64_t save_tag) {
        auto document = make_empty_document(project_uuid, 100);
        require_status(document->edit_scene_graph().upsert_node(
            SceneNodeRecord{
                .uuid = splat_uuid,
                .type = "splat",
                .name = "Splat",
                .child_order = 0,
                .payload =
                    PayloadBinding{
                        .fourcc = "SPLT",
                        .instance_uuid = splat_uuid,
                        .source_kind = "ply",
                    },
            }));
        auto splat = require_result(SplatChapterPayload::capture(
            model, SplatSourceKind::ImportedPly, false));
        require_status(document->set_splat(splat_uuid, std::move(splat)));
        require_status(document->edit_project().upsert_embed_decision(
            EmbedDecision{
                .uuid = splat_uuid,
                .node_uuid = splat_uuid,
                .payload_fourcc = "SPLT",
                .decision = "embedded",
                .reason = "stream-hydrate fixture",
            }));
        require_status(document->edit_project().upsert_embedded_payload_provenance(
            provenance(splat_uuid, "SPLT", "assets/SPLT.bin", 41)));
        (void)require_result(document->save(path, save_options(save_tag, 200)));
    }

    void rewrite_splat_chunk(const fs::path& path, const Uuid& splat_uuid,
                             const Compression compression,
                             const std::uint64_t commit_tag,
                             const bool block_crcs = false) {
        ProjectReader seed = require_result(ProjectReader::open(path));
        const ChunkInfo* row = seed.find(FOURCC_SPLT, splat_uuid);
        ASSERT_NE(row, nullptr);
        auto bytes = require_result(seed.read_chunk(*row));
        ProjectWriter writer = require_result(ProjectWriter::append(
            path,
            AppendOptions{
                .index_compression =
                    IndexCompression::StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            }));
        require_status(writer.plan_commit(CommitOptions{
            .kind = CommitKind::Explicit,
            .commit_uuid = fixed_uuid(commit_tag),
            .snapshot_uuid = seed.commit().snapshot_uuid,
            .wallclock_unix_ns = 300,
        }));
        require_status(writer.preflight(bytes.size()));
        for (const auto& chunk : seed.chunks()) {
            if (!chunk.is_live()) {
                continue;
            }
            if (chunk.key.fourcc == FOURCC_SPLT &&
                chunk.key.instance_uuid == splat_uuid) {
                require_status(writer.write_chunk(
                    chunk.key, bytes,
                    ChunkWriteOptions{
                        .chunk_version = chunk.chunk_version,
                        .compression = compression,
                        .tensor_payload = true,
                        .block_crcs = block_crcs ||
                                      chunk.block_crc_table.has_value(),
                    }));
                continue;
            }
            auto proof = require_result(seed.make_clean_proof(chunk, 1));
            require_status(writer.reuse_if_clean(proof, 1));
        }
        require_status(writer.commit());
    }

    void expect_splats_bit_equal(const lfs::core::SplatData& lhs,
                                 const lfs::core::SplatData& rhs) {
        EXPECT_EQ(lhs.get_active_sh_degree(), rhs.get_active_sh_degree());
        EXPECT_EQ(lhs.get_max_sh_degree(), rhs.get_max_sh_degree());
        EXPECT_EQ(lhs.get_scene_scale(), rhs.get_scene_scale());
        EXPECT_EQ(lhs.size(), rhs.size());
        EXPECT_EQ(tensor_bytes(lhs.means()), tensor_bytes(rhs.means()));
        EXPECT_EQ(tensor_bytes(lhs.scaling_raw()),
                  tensor_bytes(rhs.scaling_raw()));
        EXPECT_EQ(tensor_bytes(lhs.rotation_raw()),
                  tensor_bytes(rhs.rotation_raw()));
        EXPECT_EQ(tensor_bytes(lhs.opacity_raw()),
                  tensor_bytes(rhs.opacity_raw()));
        EXPECT_EQ(tensor_bytes(lhs.sh0()), tensor_bytes(rhs.sh0()));
        EXPECT_EQ(tensor_bytes(lhs.shN()), tensor_bytes(rhs.shN()));
        EXPECT_EQ(lhs.has_deleted_mask(), rhs.has_deleted_mask());
        if (lhs.has_deleted_mask() && rhs.has_deleted_mask()) {
            EXPECT_EQ(tensor_bytes(lhs.deleted()), tensor_bytes(rhs.deleted()));
        }
        ASSERT_EQ(lhs.frozen_ranges().size(), rhs.frozen_ranges().size());
        for (std::size_t i = 0; i < lhs.frozen_ranges().size(); ++i) {
            EXPECT_EQ(lhs.frozen_ranges()[i].start, rhs.frozen_ranges()[i].start);
            EXPECT_EQ(lhs.frozen_ranges()[i].count, rhs.frozen_ranges()[i].count);
        }
    }

    Hash128 splat_provenance_hash(const ProjectDocument& document,
                                  const Uuid& splat_uuid) {
        const auto records =
            require_result(document.project().embedded_payload_provenance());
        for (const auto& record : records) {
            if (record.node_uuid == splat_uuid && record.fourcc == "SPLT") {
                return record.content_xxh3_128;
            }
        }
        ADD_FAILURE() << "missing SPLT provenance for " << splat_uuid.to_string();
        return {};
    }

    struct PayloadCapGuard {
        explicit PayloadCapGuard(const std::uint64_t value) {
            detail::set_max_payload_materialized_bytes_for_testing(value);
        }
        PayloadCapGuard(const PayloadCapGuard&) = delete;
        PayloadCapGuard& operator=(const PayloadCapGuard&) = delete;
        ~PayloadCapGuard() {
            detail::set_max_payload_materialized_bytes_for_testing(std::nullopt);
        }
    };

    TEST(SplatChapterTest,
         ImportedBytesRoundTripAndTrainingOrLiveRadEmbeddingIsRefused) {
        auto model = make_splat(3);
        auto imported = SplatChapterPayload::capture(
            *model, SplatSourceKind::ImportedPly, false);
        ASSERT_TRUE(imported)
            << lfs::format_for_developer(imported.error());
        EXPECT_EQ(imported->lfsp_version(), 2u);

        // SPZ imports use ImportedSpz (not ImportedPly); embed is allowed.
        auto imported_spz = SplatChapterPayload::capture(
            *model, SplatSourceKind::ImportedSpz, false);
        ASSERT_TRUE(imported_spz)
            << lfs::format_for_developer(imported_spz.error());
        EXPECT_EQ(imported_spz->lfsp_version(), 2u);
        EXPECT_TRUE(std::ranges::equal(imported_spz->bytes(),
                                       imported->bytes()));

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

    TEST(SpanStreambufTest, WindowedReadsAndSeeks) {
        constexpr std::size_t n = 1000;
        std::vector<std::byte> pattern(n);
        for (std::size_t i = 0; i < n; ++i) {
            pattern[i] = static_cast<std::byte>((i * 131u) & 0xffu);
        }
        SpanStreambuf buffer(
            std::span<const std::byte>(pattern.data(),
                                       pattern.size()),
            7);
        std::istream stream(&buffer);

        for (std::size_t i = 0; i < n; ++i) {
            const int ch = stream.get();
            ASSERT_NE(ch, std::char_traits<char>::eof()) << i;
            EXPECT_EQ(static_cast<unsigned char>(ch),
                      static_cast<unsigned char>(pattern[i]))
                << i;
        }
        EXPECT_EQ(stream.get(), std::char_traits<char>::eof());

        stream.clear();
        stream.seekg(0);
        ASSERT_TRUE(stream);
        std::array<char, 64> chunk{};
        stream.read(chunk.data(), 64);
        ASSERT_EQ(stream.gcount(), 64);
        ASSERT_TRUE(stream);
        for (std::size_t i = 0; i < chunk.size(); ++i) {
            EXPECT_EQ(static_cast<unsigned char>(chunk[i]),
                      static_cast<unsigned char>(pattern[i]))
                << i;
        }

        stream.clear();
        stream.seekg(333, std::ios::beg);
        EXPECT_EQ(stream.tellg(), std::streampos(333));
        ASSERT_TRUE(stream);
        {
            const int ch = stream.get();
            ASSERT_NE(ch, std::char_traits<char>::eof());
            EXPECT_EQ(static_cast<unsigned char>(ch),
                      static_cast<unsigned char>(pattern[333]));
        }

        stream.seekg(17, std::ios::cur);
        EXPECT_EQ(stream.tellg(), std::streampos(351));
        ASSERT_TRUE(stream);
        {
            const int ch = stream.get();
            ASSERT_NE(ch, std::char_traits<char>::eof());
            EXPECT_EQ(static_cast<unsigned char>(ch),
                      static_cast<unsigned char>(pattern[351]));
        }

        stream.seekg(-12, std::ios::cur);
        EXPECT_EQ(stream.tellg(), std::streampos(340));
        ASSERT_TRUE(stream);
        {
            const int ch = stream.get();
            ASSERT_NE(ch, std::char_traits<char>::eof());
            EXPECT_EQ(static_cast<unsigned char>(ch),
                      static_cast<unsigned char>(pattern[340]));
        }

        stream.seekg(-80, std::ios::end);
        EXPECT_EQ(stream.tellg(), std::streampos(920));
        ASSERT_TRUE(stream);
        {
            const int ch = stream.get();
            ASSERT_NE(ch, std::char_traits<char>::eof());
            EXPECT_EQ(static_cast<unsigned char>(ch),
                      static_cast<unsigned char>(pattern[920]));
        }

        stream.clear();
        stream.seekg(static_cast<std::streamoff>(n),
                     std::ios::beg);
        ASSERT_TRUE(stream);
        EXPECT_EQ(stream.tellg(), std::streampos(n));
        EXPECT_EQ(stream.get(), std::char_traits<char>::eof());
        EXPECT_TRUE(stream.eof());

        stream.clear();
        stream.seekg(static_cast<std::streamoff>(n + 1),
                     std::ios::beg);
        EXPECT_TRUE(stream.fail());

        stream.clear();
        stream.seekg(0);
        ASSERT_TRUE(stream);
        stream.seekg(-1, std::ios::beg);
        EXPECT_TRUE(stream.fail());

        stream.clear();
        stream.seekg(static_cast<std::streamoff>(n - 10));
        ASSERT_TRUE(stream);
        std::array<char, 32> tail{};
        stream.read(tail.data(), 32);
        EXPECT_EQ(stream.gcount(), 10);
        EXPECT_TRUE(stream.eof());
        for (std::size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(static_cast<unsigned char>(tail[i]),
                      static_cast<unsigned char>(
                          pattern[n - 10 + i]))
                << i;
        }
    }

    TEST(ProjectDocumentTest, VisitStreamOwnedSeekableRoundTrip) {
        constexpr std::size_t n = 512;
        std::vector<std::byte> pattern(n);
        for (std::size_t i = 0; i < n; ++i) {
            pattern[i] = static_cast<std::byte>((i * 131u) & 0xffu);
        }
        const auto expected = pattern;
        auto chunk = require_result(LazyChunkValue::from_owned(
            std::move(pattern), fixed_uuid(1679)));
        auto visited = chunk.visit_stream(
            [&](std::istream& stream,
                const std::uint64_t bytes) -> lfs::Result<void> {
                EXPECT_EQ(bytes, n);
                std::array<char, 32> head{};
                stream.read(
                    head.data(),
                    static_cast<std::streamsize>(head.size()));
                EXPECT_TRUE(stream.good());
                EXPECT_EQ(stream.gcount(), 32);
                for (std::size_t i = 0; i < head.size(); ++i) {
                    EXPECT_EQ(
                        static_cast<unsigned char>(head[i]),
                        static_cast<unsigned char>(expected[i]))
                        << i;
                }
                const auto tail_offset =
                    static_cast<std::streamoff>(bytes - 16);
                stream.seekg(tail_offset);
                EXPECT_TRUE(stream.good());
                EXPECT_EQ(stream.tellg(),
                          std::streampos(tail_offset));
                std::array<char, 16> tail{};
                stream.read(
                    tail.data(),
                    static_cast<std::streamsize>(tail.size()));
                EXPECT_TRUE(stream.good());
                EXPECT_EQ(stream.gcount(), 16);
                for (std::size_t i = 0; i < tail.size(); ++i) {
                    EXPECT_EQ(
                        static_cast<unsigned char>(tail[i]),
                        static_cast<unsigned char>(
                            expected[expected.size() - 16 + i]))
                        << i;
                }
                return {};
            });
        ASSERT_TRUE(visited)
            << lfs::format_for_developer(visited.error());
    }

    TEST(SpanStreambufTest, SpanStreambufServesBuffersOver2GiB) {
        const auto int_max = static_cast<std::size_t>(
            std::numeric_limits<int>::max());
        const std::size_t size = int_max + 4096;
        std::vector<std::byte> bytes(size, std::byte{0});
        for (std::size_t i = 0; i < 40; ++i) {
            bytes[i] = static_cast<std::byte>(0xA0 + i);
        }
        for (std::size_t i = 0; i < 16; ++i) {
            bytes[int_max - 8 + i] =
                static_cast<std::byte>(0xB0 + i);
        }
        for (std::size_t i = 0; i < 16; ++i) {
            bytes[size - 16 + i] =
                static_cast<std::byte>(0xC0 + i);
        }

        SpanStreambuf buffer(
            std::span<const std::byte>(bytes.data(),
                                       bytes.size()));
        std::istream stream(&buffer);

        std::array<char, 40> head{};
        stream.read(head.data(), 40);
        ASSERT_TRUE(stream.good());
        ASSERT_EQ(stream.gcount(), 40);
        for (std::size_t i = 0; i < head.size(); ++i) {
            EXPECT_EQ(static_cast<unsigned char>(head[i]),
                      static_cast<unsigned char>(bytes[i]))
                << i;
        }

        stream.seekg(static_cast<std::streamoff>(int_max - 8));
        ASSERT_TRUE(stream.good());
        std::array<char, 16> mid{};
        stream.read(mid.data(), 16);
        ASSERT_TRUE(stream.good());
        ASSERT_EQ(stream.gcount(), 16);
        for (std::size_t i = 0; i < mid.size(); ++i) {
            EXPECT_EQ(
                static_cast<unsigned char>(mid[i]),
                static_cast<unsigned char>(bytes[int_max - 8 + i]))
                << i;
        }

        stream.seekg(-16, std::ios::end);
        ASSERT_TRUE(stream.good());
        std::array<char, 16> tail{};
        stream.read(tail.data(), 16);
        ASSERT_TRUE(stream.good());
        ASSERT_EQ(stream.gcount(), 16);
        for (std::size_t i = 0; i < tail.size(); ++i) {
            EXPECT_EQ(
                static_cast<unsigned char>(tail[i]),
                static_cast<unsigned char>(bytes[size - 16 + i]))
                << i;
        }
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

    TEST(SceneChapterAdapterTest,
         ChildOrderSortsSiblingsWhenDomArrayIsShuffled) {
        SceneGraphChapter chapter;
        const Uuid root = fixed_uuid(930);
        const Uuid first = fixed_uuid(931);
        const Uuid second = fixed_uuid(932);
        const Uuid third = fixed_uuid(933);
        // Intentionally upsert in reverse sibling order; child_order is
        // authoritative for restore insertion order.
        require_status(chapter.upsert_node(SceneNodeRecord{
            .uuid = root,
            .type = "group",
            .name = "Root",
            .child_order = 0,
        }));
        require_status(chapter.upsert_node(SceneNodeRecord{
            .uuid = third,
            .type = "group",
            .name = "Third",
            .parent_uuid = root,
            .child_order = 2,
        }));
        require_status(chapter.upsert_node(SceneNodeRecord{
            .uuid = first,
            .type = "group",
            .name = "First",
            .parent_uuid = root,
            .child_order = 0,
        }));
        require_status(chapter.upsert_node(SceneNodeRecord{
            .uuid = second,
            .type = "group",
            .name = "Second",
            .parent_uuid = root,
            .child_order = 1,
        }));

        Scene live;
        auto restored =
            hydrate_scene_graph(chapter, live, ScenePayloadResolver{});
        ASSERT_TRUE(restored)
            << lfs::format_for_developer(restored.error());
        const auto* root_node = live.getNodeByUuid(root);
        ASSERT_NE(root_node, nullptr);
        ASSERT_EQ(root_node->children.size(), 3u);
        EXPECT_EQ(live.getNodeUuid(root_node->children[0]), first);
        EXPECT_EQ(live.getNodeUuid(root_node->children[1]), second);
        EXPECT_EQ(live.getNodeUuid(root_node->children[2]), third);
    }

    TEST(ProjectDocumentTest,
         SaveReportReusesRefreshedSourceReader) {
        TemporaryDirectory temporary;
        const auto path =
            temporary.path / "save-report-reader.licht";
        auto document =
            make_empty_document(fixed_uuid(911), 100);
        const auto report = require_result(
            document->save(path, save_options(911, 200)));
        ASSERT_NE(document->source_reader(), nullptr);
        ASSERT_TRUE(document->source_commit_uuid());
        EXPECT_EQ(
            *document->source_commit_uuid(),
            report.commit_uuid);
        EXPECT_EQ(
            document->generation(),
            report.generation);
        EXPECT_EQ(
            document->source_reader()->commit().generation,
            report.generation);
        EXPECT_EQ(
            document->source_reader()->commit().snapshot_uuid,
            report.snapshot_uuid);
    }

    TEST(ProjectDocumentTest,
         LegacyGuiWindowIsDiscardedWithoutDirtyingAndRemovedOnSave) {
        TemporaryDirectory temporary;
        const auto path =
            temporary.path / "legacy-gui-window.licht";
        auto document =
            make_empty_document(fixed_uuid(912), 100);
        require_status(document->edit_view().dom().set(
            "tools.active_tool_id",
            std::string{"builtin.select"}));
        (void)require_result(
            document->save(path, save_options(1912, 200)));
        inject_legacy_gui_window(path, 1913, 300);

        auto reopened = require_result_ptr(
            ProjectDocument::open(path));
        EXPECT_FALSE(reopened->dirty());
        EXPECT_TRUE(reopened->dirty_chapters().empty());
        const auto loaded_tool =
            reopened->view().dom().get<std::string>(
                "tools.active_tool_id");
        ASSERT_TRUE(loaded_tool.has_value());
        EXPECT_EQ(*loaded_tool, "builtin.select");
        const auto sanitized_gui =
            json_root(reopened->gui_layout().dom());
        EXPECT_FALSE(
            sanitized_gui["layouts"][0]["areas"][0]
                         ["spaces"][0]["opaque_payload"]
                             .contains("window"));

        (void)require_result(
            reopened->save(path, save_options(1914, 400)));
        auto reader = require_result(
            ProjectReader::open(path));
        const auto* gui_row = reader.find(
            FOURCC_GUIL, reader.superblock().project_uuid);
        ASSERT_NE(gui_row, nullptr);
        auto raw_gui = require_result(
            reader.read_chunk(*gui_row));
        auto raw_gui_dom = require_result(lfs::io::JsonChapterDom::from_bytes(raw_gui));
        const auto persisted_gui = json_root(raw_gui_dom);
        EXPECT_FALSE(
            persisted_gui["layouts"][0]["areas"][0]
                         ["spaces"][0]["opaque_payload"]
                             .contains("window"));

        auto saved_again = require_result_ptr(
            ProjectDocument::open(path));
        const auto saved_tool =
            saved_again->view().dom().get<std::string>(
                "tools.active_tool_id");
        ASSERT_TRUE(saved_tool.has_value());
        EXPECT_EQ(*saved_tool, "builtin.select");
    }

    TEST(ProjectDocumentTest,
         LegacyGuiWindowNormalizationSurvivesSaveAsRebind) {
        TemporaryDirectory temporary;
        const auto source =
            temporary.path / "legacy-gui-window-source.licht";
        const auto destination =
            temporary.path / "legacy-gui-window-save-as.licht";
        auto document =
            make_empty_document(fixed_uuid(913), 100);
        require_status(document->edit_view().dom().set(
            "tools.active_tool_id",
            std::string{"builtin.select"}));
        (void)require_result(
            document->save(source, save_options(1915, 200)));
        inject_legacy_gui_window(source, 1916, 300);

        auto reopened = require_result_ptr(
            ProjectDocument::open(source));
        EXPECT_FALSE(reopened->dirty());
        EXPECT_TRUE(reopened->dirty_chapters().empty());

        auto saved = reopened->save_as(
            destination, save_options(1917, 400));
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(saved.error());
        ASSERT_TRUE(reopened->source_path().has_value());
        EXPECT_EQ(
            *reopened->source_path(),
            std::filesystem::absolute(destination)
                .lexically_normal());

        auto saved_as = require_result_ptr(
            ProjectDocument::open(destination));
        EXPECT_FALSE(saved_as->dirty());
        EXPECT_TRUE(saved_as->dirty_chapters().empty());
        const auto saved_tool =
            saved_as->view().dom().get<std::string>(
                "tools.active_tool_id");
        ASSERT_TRUE(saved_tool.has_value());
        EXPECT_EQ(*saved_tool, "builtin.select");
        const auto persisted_gui =
            json_root(saved_as->gui_layout().dom());
        EXPECT_FALSE(
            persisted_gui["layouts"][0]["areas"][0]
                         ["spaces"][0]["opaque_payload"]
                             .contains("window"));
    }

    TEST(ProjectDocumentTest,
         InvalidPendingParametersRefuseHydrationBeforeSceneMutation) {
        auto document = make_empty_document(fixed_uuid(910), 100);
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
         SelectionReferencingMissingSceneNodeRefusesHydration) {
        auto document = make_empty_document(fixed_uuid(920), 100);
        require_status(document->edit_selection().set_selected_node_uuids(
            {fixed_uuid(921)}));

        Scene live;
        ASSERT_NE(live.addGroup("Existing"), lfs::core::NULL_NODE);
        const auto before = witness_scene(live);
        auto staged = document->stage_hydration(live);
        ASSERT_FALSE(staged);
        EXPECT_EQ(staged.error().code(), lfs::ErrorCode::DataLoss);
        EXPECT_EQ(witness_scene(live), before);
    }

    TEST(ProjectDocumentTest,
         SelectedGeneratedKeyframeNodesDoNotSurviveSaveAndReopen) {
        Scene source;
        const auto group_id = source.addGroup("Persistent group");
        const auto keyframe_group_id =
            source.addKeyframeGroup("Generated keyframes");
        const auto keyframe_id = source.addKeyframe(
            "Generated keyframe", keyframe_group_id,
            std::make_unique<lfs::core::KeyframeData>());
        ASSERT_NE(group_id, lfs::core::NULL_NODE);
        ASSERT_NE(keyframe_group_id, lfs::core::NULL_NODE);
        ASSERT_NE(keyframe_id, lfs::core::NULL_NODE);

        const auto group_uuid = source.getNodeUuid(group_id);
        const auto keyframe_group_uuid =
            source.getNodeUuid(keyframe_group_id);
        const auto keyframe_uuid = source.getNodeUuid(keyframe_id);
        auto scene_chapter =
            capture_scene_graph(source, ScenePayloadBindings{});
        ASSERT_TRUE(scene_chapter)
            << lfs::format_for_developer(scene_chapter.error());
        const std::array selected{
            group_uuid, keyframe_group_uuid, keyframe_uuid};
        auto selection_chapter =
            capture_selection_chapter(source, selected);
        ASSERT_TRUE(selection_chapter)
            << lfs::format_for_developer(selection_chapter.error());
        EXPECT_EQ(selection_chapter->selected_node_uuids(),
                  (std::vector<Uuid>{group_uuid}));

        auto document = make_empty_document(fixed_uuid(923), 100);
        document->edit_scene_graph() = std::move(*scene_chapter);
        document->edit_selection() = std::move(*selection_chapter);
        TemporaryDirectory temporary;
        const auto path = temporary.path / "keyframe-selection.licht";
        (void)require_result(document->save(path, save_options(923, 200)));

        auto reopened = require_result_ptr(ProjectDocument::open(path));
        EXPECT_EQ(reopened->selection().selected_node_uuids(),
                  (std::vector<Uuid>{group_uuid}));
        Scene restored;
        auto report = reopened->hydrate(restored);
        ASSERT_TRUE(report)
            << lfs::format_for_developer(report.error());
        EXPECT_EQ(report->selection.selected_node_uuids,
                  (std::vector<Uuid>{group_uuid}));
        EXPECT_NE(restored.getNodeByUuid(group_uuid), nullptr);
        EXPECT_EQ(restored.getNodeByUuid(keyframe_group_uuid), nullptr);
        EXPECT_EQ(restored.getNodeByUuid(keyframe_uuid), nullptr);
    }

    TEST(ProjectDocumentTest,
         TruncatedStandalonePpispRefusesHydrationBeforeSceneMutation) {
        const auto payload_uuid = fixed_uuid(925);
        auto document = make_empty_document(fixed_uuid(924), 100);
        auto bytes = std::make_shared<const std::vector<std::byte>>(
            std::vector<std::byte>{std::byte{0x50}, std::byte{0x50}});
        auto payload = LazyChunkValue::from_owned(bytes, payload_uuid);
        ASSERT_TRUE(payload)
            << lfs::format_for_developer(payload.error());
        require_status(document->set_ppisp(
            payload_uuid, std::move(*payload)));

        Scene live;
        ASSERT_NE(live.addGroup("Existing"), lfs::core::NULL_NODE);
        const auto before = witness_scene(live);
        auto staged = document->stage_hydration(live);
        ASSERT_FALSE(staged);
        EXPECT_EQ(staged.error().code(), lfs::ErrorCode::DataLoss);
        EXPECT_EQ(witness_scene(live), before);
    }

    TEST(ProjectDocumentTest,
         CheckpointWindowPastStoredPayloadRefusesHydrationBeforeSceneMutation) {
        const auto training_uuid = fixed_uuid(926);
        const auto checkpoint_uuid = fixed_uuid(927);
        auto model = make_splat(2);
        lfs::training::MCMC strategy(*model);
        lfs::core::param::TrainingParameters parameters;
        parameters.optimization =
            lfs::core::param::OptimizationParameters::mcmc_defaults();
        parameters.optimization.sh_degree = 0;
        parameters.optimization.max_cap = 2;
        strategy.initialize(parameters.optimization);

        std::ostringstream stream(std::ios::binary | std::ios::out);
        auto serialized = lfs::training::serialize_checkpoint(
            stream, 11, strategy, parameters, nullptr, nullptr, nullptr,
            nullptr);
        ASSERT_TRUE(serialized)
            << lfs::format_for_developer(serialized.error());
        const auto encoded = stream.str();
        ASSERT_GT(encoded.size(), 32u);
        std::vector<std::byte> truncated(encoded.size() - 1);
        std::memcpy(truncated.data(), encoded.data(), truncated.size());

        auto document = make_empty_document(fixed_uuid(928), 100);
        require_status(document->edit_scene_graph().upsert_node(
            SceneNodeRecord{
                .uuid = training_uuid,
                .type = "splat",
                .name = "Training",
                .child_order = 0,
                .payload = PayloadBinding{
                    .fourcc = "CKPT",
                    .instance_uuid = checkpoint_uuid,
                    .source_kind = "training",
                },
            }));
        require_status(document->edit_scene_graph().set_training_model_uuid(
            training_uuid));
        auto payload = LazyChunkValue::from_owned(
            std::move(truncated), checkpoint_uuid);
        ASSERT_TRUE(payload)
            << lfs::format_for_developer(payload.error());
        require_status(document->set_checkpoint(
            checkpoint_uuid, std::move(*payload)));

        Scene live;
        ASSERT_NE(live.addGroup("Existing"), lfs::core::NULL_NODE);
        const auto before = witness_scene(live);
        auto staged = document->stage_hydration(live);
        ASSERT_FALSE(staged);
        EXPECT_EQ(staged.error().code(), lfs::ErrorCode::DataLoss);
        EXPECT_EQ(witness_scene(live), before);
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
            auto document = require_result_ptr(ProjectDocument::open(path));
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
         DeferredShellExposesUnloadedUnitsAndCommitsHydrationPerNodeUuid) {
        TemporaryDirectory temporary;
        const auto path =
            temporary.path / "partial-hydration.licht";
        write_phase_a_fixture(path);

        auto document = require_result_ptr(ProjectDocument::open(
            path,
            ProjectDocumentOpenOptions{
                .defer_geometry_payloads = true,
            }));
        ASSERT_EQ(document->payload_states().size(), 3u);
        EXPECT_TRUE(std::ranges::all_of(
            document->payload_states(),
            [](const auto& state) {
                return !state.loaded;
            }));

        Scene live;
        auto shell = document->stage_shell(live);
        ASSERT_TRUE(shell)
            << lfs::format_for_developer(
                   shell.error());
        live.commitRestoreStage(std::move(*shell));
        ASSERT_EQ(live.getSelectionGroups().size(), 1u);
        EXPECT_EQ(live.getSelectionGroups().front().id, 4u);
        EXPECT_EQ(
            live.getSelectionGroups().front().name,
            "Fixture");
        EXPECT_EQ(live.getActiveSelectionGroup(), 4u);
        EXPECT_EQ(live.getSelectionMask(), nullptr);
        for (const auto uuid :
             {fixed_uuid(952), fixed_uuid(953),
              fixed_uuid(954)}) {
            const auto* node =
                live.getNodeByUuid(uuid);
            ASSERT_NE(node, nullptr);
            EXPECT_EQ(
                node->payload_hydration,
                lfs::core::
                    PayloadHydrationState::
                        Unloaded);
        }

        auto staged =
            document->stage_hydration(live);
        ASSERT_TRUE(staged)
            << lfs::format_for_developer(
                   staged.error());

        glm::mat4 edited_transform{1.0f};
        edited_transform[3] =
            glm::vec4(7.0f, 8.0f, 9.0f, 1.0f);
        live.setNodeTransform(
            live.getNodeIdByUuid(fixed_uuid(952)),
            edited_transform);
        live.removeNodeById(
            live.getNodeIdByUuid(fixed_uuid(953)));

        const auto report =
            ProjectDocument::commit_partial_hydration(
                live, std::move(*staged), true);
        EXPECT_EQ(report.hydrated_payload_units, 2u);
        EXPECT_EQ(report.invalidated_payload_units, 1u);
        EXPECT_FALSE(report.selection_installed);
        EXPECT_EQ(
            live.getNodeIdByUuid(fixed_uuid(953)),
            lfs::core::NULL_NODE);

        const auto* splat =
            live.getNodeByUuid(fixed_uuid(952));
        ASSERT_NE(splat, nullptr);
        ASSERT_NE(splat->model, nullptr);
        EXPECT_EQ(splat->model->size(), 3);
        EXPECT_EQ(
            splat->payload_hydration,
            lfs::core::PayloadHydrationState::Loaded);
        EXPECT_EQ(
            splat->local_transform.get(),
            edited_transform);

        const auto* mesh =
            live.getNodeByUuid(fixed_uuid(954));
        ASSERT_NE(mesh, nullptr);
        ASSERT_NE(mesh->mesh, nullptr);
        EXPECT_EQ(
            mesh->payload_hydration,
            lfs::core::PayloadHydrationState::Loaded);
    }

    TEST(ProjectDocumentTest,
         SaveBeforeHydrationCarriesEveryUnloadedPayloadSpanForward) {
        TemporaryDirectory temporary;
        const auto path =
            temporary.path / "partial-save.licht";
        write_phase_a_fixture(path);

        auto first_reader =
            ProjectReader::open(path);
        ASSERT_TRUE(first_reader)
            << lfs::format_for_developer(
                   first_reader.error());
        const std::array payload_keys{
            ChunkKey{FOURCC_SPLT, fixed_uuid(952)},
            ChunkKey{FOURCC_PCLD, fixed_uuid(953)},
            ChunkKey{FOURCC_MESH, fixed_uuid(954)},
        };
        std::array<ChunkInfo, 3> first_rows;
        std::array<std::vector<std::byte>, 3>
            first_payloads;
        for (std::size_t index = 0;
             index < payload_keys.size(); ++index) {
            const auto* row = first_reader->find(
                payload_keys[index].fourcc,
                payload_keys[index].instance_uuid);
            ASSERT_NE(row, nullptr);
            first_rows[index] = *row;
            auto bytes =
                first_reader->read_chunk(*row);
            ASSERT_TRUE(bytes);
            first_payloads[index] =
                std::move(*bytes);
        }

        auto document = require_result_ptr(ProjectDocument::open(
            path,
            ProjectDocumentOpenOptions{
                .defer_geometry_payloads = true,
            }));
        document->edit_metrics()
            .accumulated_training_seconds = 1.0;
        auto saved =
            document->save(
                path, save_options(980, 300));
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(
                   saved.error());
        EXPECT_EQ(saved->generation, 2u);

        auto second_reader =
            ProjectReader::open(path);
        ASSERT_TRUE(second_reader)
            << lfs::format_for_developer(
                   second_reader.error());
        for (std::size_t index = 0;
             index < payload_keys.size(); ++index) {
            const auto* row = second_reader->find(
                payload_keys[index].fourcc,
                payload_keys[index].instance_uuid);
            ASSERT_NE(row, nullptr);
            EXPECT_EQ(
                row->header_offset,
                first_rows[index].header_offset);
            EXPECT_EQ(row->source_generation, 1u);
            auto bytes =
                second_reader->read_chunk(*row);
            ASSERT_TRUE(bytes);
            EXPECT_EQ(*bytes, first_payloads[index]);
        }
    }

    TEST(ProjectDocumentTest,
         ExplicitPreviewRegeneratesWhileAutomaticSaveCarriesItForward) {
        TemporaryDirectory temporary;
        const auto path =
            temporary.path / "preview-policy.licht";
        auto document = make_empty_document(fixed_uuid(990), 100);
        const auto preview = one_pixel_png();
        auto first_options =
            save_options(991, 200);
        first_options.preview_png =
            std::span<const std::byte>(preview);
        auto first =
            document->save(path, first_options);
        ASSERT_TRUE(first)
            << lfs::format_for_developer(
                   first.error());

        auto first_reader =
            ProjectReader::open(path);
        ASSERT_TRUE(first_reader);
        ASSERT_TRUE(first_reader->preview());
        const auto first_locator =
            *first_reader->preview();
        auto first_bytes =
            first_reader->read_preview();
        ASSERT_TRUE(first_bytes);
        EXPECT_EQ(*first_bytes, preview);

        auto reopened = require_result_ptr(ProjectDocument::open(path));
        reopened->edit_metrics()
            .accumulated_training_seconds = 1.0;
        auto second =
            reopened->save(
                path, save_options(994, 300));
        ASSERT_TRUE(second)
            << lfs::format_for_developer(
                   second.error());
        auto second_reader =
            ProjectReader::open(path);
        ASSERT_TRUE(second_reader);
        ASSERT_TRUE(second_reader->preview());
        EXPECT_EQ(
            *second_reader->preview(),
            first_locator);
        auto second_bytes =
            second_reader->read_preview();
        ASSERT_TRUE(second_bytes);
        EXPECT_EQ(*second_bytes, preview);

        auto invalid_options =
            save_options(997, 400);
        invalid_options.commit.kind =
            CommitKind::Autosave;
        invalid_options.preview_png =
            std::span<const std::byte>(preview);
        auto invalid =
            reopened->save(
                path, invalid_options);
        ASSERT_FALSE(invalid);
        EXPECT_EQ(
            invalid.error().code(),
            lfs::ErrorCode::
                FailedPrecondition);
    }

    TEST(ProjectDocumentTest,
         RemovingUnloadedNodePayloadCreatesADeletionTombstone) {
        TemporaryDirectory temporary;
        const auto path =
            temporary.path / "partial-delete.licht";
        write_phase_a_fixture(path);
        auto document = require_result_ptr(ProjectDocument::open(
            path,
            ProjectDocumentOpenOptions{
                .defer_geometry_payloads = true,
            }));

        EXPECT_TRUE(
            document->remove_splat(
                fixed_uuid(952)));
        auto removed_node =
            document->edit_scene_graph()
                .remove_node(fixed_uuid(952));
        ASSERT_TRUE(removed_node);
        EXPECT_TRUE(*removed_node);
        auto point_node =
            document->edit_scene_graph().find(
                fixed_uuid(953));
        ASSERT_TRUE(point_node);
        ASSERT_TRUE(*point_node);
        point_node->value().child_order = 0;
        require_status(
            document->edit_scene_graph().upsert_node(
                point_node->value()));
        auto mesh_node =
            document->edit_scene_graph().find(
                fixed_uuid(954));
        ASSERT_TRUE(mesh_node);
        ASSERT_TRUE(*mesh_node);
        mesh_node->value().child_order = 1;
        require_status(
            document->edit_scene_graph().upsert_node(
                mesh_node->value()));
        EXPECT_TRUE(document->dirty());
        const auto dirty_chapters =
            document->dirty_chapters();
        EXPECT_NE(
            std::ranges::find(
                dirty_chapters,
                "SPLT"),
            dirty_chapters.end());
        auto saved =
            document->save(
                path, save_options(1000, 500));
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(
                   saved.error());
        auto reader = ProjectReader::open(path);
        ASSERT_TRUE(reader);
        const auto* deleted =
            reader->find(
                FOURCC_SPLT, fixed_uuid(952));
        ASSERT_NE(deleted, nullptr);
        EXPECT_EQ(
            deleted->row_kind,
            RowKind::Tombstone);
        EXPECT_NE(
            reader->find(
                FOURCC_PCLD, fixed_uuid(953)),
            nullptr);
        auto reopened = require_result_ptr(ProjectDocument::open(path));
        const auto reopened_splats =
            reopened->splat_uuids();
        EXPECT_EQ(
            std::ranges::find(
                reopened_splats,
                fixed_uuid(952)),
            reopened_splats.end());
    }

    TEST(ProjectDocumentTest,
         WholesaleSceneReplaceDropsOrphanDeferredPcldOnReconcile) {
        // Catches trainer snapshot append: wholesale SCNG assignment leaves
        // a deferred PCLD key unbound. Without
        // remove_geometry_payloads_not_bound_by_scene(), save fails DataLoss
        // at the deferred-key owner check.
        TemporaryDirectory temporary;
        const auto path = temporary.path / "orphan-pcld.licht";
        const Uuid project_uuid = fixed_uuid(4401);
        const Uuid point_uuid = fixed_uuid(4402);
        const Uuid replacement_uuid = fixed_uuid(4403);

        auto document = make_empty_document(project_uuid, 100);
        require_status(document->edit_scene_graph().upsert_node(
            SceneNodeRecord{
                .uuid = point_uuid,
                .type = "pointcloud",
                .name = "Initial points",
                .child_order = 0,
                .payload =
                    PayloadBinding{
                        .fourcc = "PCLD",
                        .instance_uuid = point_uuid,
                        .source_kind = "ply",
                    },
            }));
        require_status(document->set_point_cloud(
            point_uuid, PointCloudPayload(make_point_cloud(2))));
        require_status(document->edit_project().upsert_embed_decision(
            EmbedDecision{
                .uuid = point_uuid,
                .node_uuid = point_uuid,
                .payload_fourcc = "PCLD",
                .decision = "embedded",
                .reason = "initial cloud",
            }));
        require_status(
            document->edit_project().upsert_embedded_payload_provenance(
                provenance(
                    point_uuid, "PCLD", "assets/points.ply", 44)));
        ASSERT_TRUE(document->save(path, save_options(4404, 200)));

        auto opened = require_result_ptr(ProjectDocument::open(
            path,
            ProjectDocumentOpenOptions{
                .defer_geometry_payloads = true,
            }));
        ASSERT_EQ(opened->point_cloud_uuids().size(), 1u);

        opened->edit_scene_graph() = SceneGraphChapter();
        require_status(opened->edit_scene_graph().upsert_node(
            SceneNodeRecord{
                .uuid = replacement_uuid,
                .type = "group",
                .name = "Training root",
                .child_order = 0,
            }));

        auto refused = opened->save(path, save_options(4405, 300));
        ASSERT_FALSE(refused);
        EXPECT_EQ(refused.error().code(), lfs::ErrorCode::DataLoss);

        opened->remove_geometry_payloads_not_bound_by_scene();
        auto saved = opened->save(path, save_options(4406, 400));
        ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());

        auto reopened = require_result_ptr(ProjectDocument::open(path));
        EXPECT_TRUE(reopened->point_cloud_uuids().empty());
        EXPECT_EQ(reopened->find_point_cloud(point_uuid), nullptr);
        auto nodes = reopened->scene_graph().nodes();
        ASSERT_TRUE(nodes);
        EXPECT_TRUE(std::ranges::none_of(
            *nodes, [](const SceneNodeRecord& node) {
                return node.payload && node.payload->fourcc == "PCLD";
            }));
        auto reader = ProjectReader::open(path);
        ASSERT_TRUE(reader);
        const auto* pcld = reader->find(FOURCC_PCLD, point_uuid);
        if (pcld != nullptr) {
            EXPECT_EQ(pcld->row_kind, RowKind::Tombstone);
        }
    }

    TEST(ProjectDocumentTest,
         WholesaleSceneReplaceConvergesDirtyEpochAfterReconcileSaveAndAdopt) {
        // GUI close-save loop: wholesale SCNG replace (training
        // removed the point-cloud node) plus reconcile+save must
        // clear PCLD dirty. An adopt-equivalent reopen and a
        // second unbound-scene replace+reconcile must not mark
        // PCLD dirty again.
        TemporaryDirectory temporary;
        const auto path =
            temporary.path / "orphan-pcld-converge.licht";
        const Uuid project_uuid = fixed_uuid(4411);
        const Uuid point_uuid = fixed_uuid(4412);
        const Uuid replacement_uuid = fixed_uuid(4413);

        const auto install_unbound_scene =
            [&](ProjectDocument& document) {
                document.edit_scene_graph() = SceneGraphChapter();
                require_status(document.edit_scene_graph().upsert_node(
                    SceneNodeRecord{
                        .uuid = replacement_uuid,
                        .type = "group",
                        .name = "Training root",
                        .child_order = 0,
                    }));
            };
        const auto dirty_has = [](const ProjectDocument& document,
                                  const std::string_view fourcc) {
            const auto chapters = document.dirty_chapters();
            return std::ranges::find(chapters, fourcc) != chapters.end();
        };

        auto document = make_empty_document(project_uuid, 100);
        require_status(document->edit_scene_graph().upsert_node(
            SceneNodeRecord{
                .uuid = point_uuid,
                .type = "pointcloud",
                .name = "Initial points",
                .child_order = 0,
                .payload =
                    PayloadBinding{
                        .fourcc = "PCLD",
                        .instance_uuid = point_uuid,
                        .source_kind = "ply",
                    },
            }));
        require_status(document->set_point_cloud(
            point_uuid, PointCloudPayload(make_point_cloud(2))));
        require_status(document->edit_project().upsert_embed_decision(
            EmbedDecision{
                .uuid = point_uuid,
                .node_uuid = point_uuid,
                .payload_fourcc = "PCLD",
                .decision = "embedded",
                .reason = "initial cloud",
            }));
        require_status(
            document->edit_project().upsert_embedded_payload_provenance(
                provenance(
                    point_uuid, "PCLD", "assets/points.ply", 45)));
        ASSERT_TRUE(document->save(path, save_options(4414, 200)));
        EXPECT_FALSE(document->dirty());

        auto opened = require_result_ptr(ProjectDocument::open(
            path,
            ProjectDocumentOpenOptions{
                .defer_geometry_payloads = true,
            }));
        ASSERT_EQ(opened->point_cloud_uuids().size(), 1u);
        install_unbound_scene(*opened);
        opened->remove_geometry_payloads_not_bound_by_scene();
        EXPECT_TRUE(dirty_has(*opened, "PCLD"));
        auto first_save = opened->save(path, save_options(4415, 300));
        ASSERT_TRUE(first_save)
            << lfs::format_for_developer(first_save.error());
        EXPECT_FALSE(opened->dirty());
        EXPECT_FALSE(dirty_has(*opened, "PCLD"));
        EXPECT_FALSE(dirty_has(*opened, "SCNG"));

        auto adopted = require_result_ptr(ProjectDocument::open(
            path,
            ProjectDocumentOpenOptions{
                .defer_geometry_payloads = true,
            }));
        EXPECT_FALSE(adopted->dirty());
        EXPECT_TRUE(adopted->point_cloud_uuids().empty());
        EXPECT_EQ(adopted->find_point_cloud(point_uuid), nullptr);

        install_unbound_scene(*adopted);
        adopted->remove_geometry_payloads_not_bound_by_scene();
        EXPECT_FALSE(dirty_has(*adopted, "PCLD"));
        auto second_save = adopted->save(path, save_options(4416, 400));
        ASSERT_TRUE(second_save)
            << lfs::format_for_developer(second_save.error());
        EXPECT_FALSE(adopted->dirty());
        EXPECT_TRUE(adopted->dirty_chapters().empty());
    }

    TEST(ProjectDocumentTest,
         SaveAsAtomicallyRebindsAndPreservesUnloadedPayloads) {
        TemporaryDirectory temporary;
        const auto source =
            temporary.path / "source.licht";
        const auto destination =
            temporary.path / "destination.licht";
        write_phase_a_fixture(source);
        {
            std::ofstream existing(
                destination,
                std::ios::binary |
                    std::ios::trunc);
            existing << "replace me";
        }

        auto document = require_result_ptr(ProjectDocument::open(
            source,
            ProjectDocumentOpenOptions{
                .defer_geometry_payloads = true,
            }));
        const auto project_uuid =
            document->project_uuid();
        auto saved = document->save_as(
            destination, save_options(1010, 600));
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(
                   saved.error());
        ASSERT_TRUE(document->source_path());
        EXPECT_EQ(
            *document->source_path(),
            std::filesystem::absolute(
                destination)
                .lexically_normal());

        auto reader =
            ProjectReader::open(destination);
        ASSERT_TRUE(reader)
            << lfs::format_for_developer(
                   reader.error());
        EXPECT_EQ(
            reader->superblock().project_uuid,
            project_uuid);
        EXPECT_NE(
            reader->find(
                FOURCC_SPLT, fixed_uuid(952)),
            nullptr);
        EXPECT_NE(
            reader->find(
                FOURCC_PCLD, fixed_uuid(953)),
            nullptr);
        EXPECT_NE(
            reader->find(
                FOURCC_MESH, fixed_uuid(954)),
            nullptr);
        EXPECT_TRUE(
            std::filesystem::is_regular_file(
                source));
    }

    TEST(ProjectDocumentTest,
         FirstSaveExplicitlyReplacesForeignDestinationAtomically) {
        TemporaryDirectory temporary;
        const auto destination = temporary.path / "destination.licht";
        auto existing = make_empty_document(fixed_uuid(970), 100);
        auto existing_options = save_options(1970, 1000);
        ASSERT_TRUE(existing->save(destination, existing_options));

        auto document = make_empty_document(fixed_uuid(971), 100);
        auto options = save_options(1971, 1100);
        options.allow_existing_destination_replacement = true;
        ASSERT_TRUE(document->save(destination, options));

        auto reader = ProjectReader::open(destination);
        ASSERT_TRUE(reader);
        EXPECT_EQ(reader->superblock().project_uuid, fixed_uuid(971));
    }

    TEST(ProjectDocumentTest,
         FirstSaveImplicitReplacementRemainsRefused) {
        TemporaryDirectory temporary;
        const auto destination = temporary.path / "destination.licht";
        auto existing = make_empty_document(fixed_uuid(972), 100);
        ASSERT_TRUE(existing->save(destination, save_options(1972, 1200)));

        auto document = make_empty_document(fixed_uuid(973), 100);
        auto refused = document->save(destination, save_options(1973, 1300));
        ASSERT_FALSE(refused);
        EXPECT_EQ(refused.error().code(), lfs::ErrorCode::AlreadyExists);

        auto reader = ProjectReader::open(destination);
        ASSERT_TRUE(reader);
        EXPECT_EQ(reader->superblock().project_uuid, fixed_uuid(972));

        // Non-default nested destinations keep the same P3 refusal.
        const auto nested =
            temporary.path / "nested" / "other.licht";
        std::filesystem::create_directories(nested.parent_path());
        auto nested_existing =
            make_empty_document(fixed_uuid(975), 100);
        ASSERT_TRUE(nested_existing->save(
            nested, save_options(1975, 1400)));
        auto nested_document =
            make_empty_document(fixed_uuid(976), 100);
        auto nested_refused = nested_document->save(
            nested, save_options(1976, 1500));
        ASSERT_FALSE(nested_refused);
        EXPECT_EQ(
            nested_refused.error().code(),
            lfs::ErrorCode::AlreadyExists);
        auto nested_reader = ProjectReader::open(nested);
        ASSERT_TRUE(nested_reader);
        EXPECT_EQ(
            nested_reader->superblock().project_uuid,
            fixed_uuid(975));
    }

    TEST(ProjectDocumentTest,
         AppendIntoForeignContainerPreservesIdentity) {
        // Missing uuid restore after overwriting PROJ with a
        // foreign chapter is refused at project_document.cpp:838
        // (DataLoss: PROJ UUID != container superblock UUID).
        TemporaryDirectory temporary;
        const auto path = temporary.path / "foreign.licht";
        const Uuid container_uuid = fixed_uuid(9801);
        const Uuid foreign_uuid = fixed_uuid(9802);
        const Uuid checkpoint_uuid = fixed_uuid(9803);
        const Uuid adopted_node = fixed_uuid(9804);

        auto existing = make_empty_document(container_uuid, 100);
        require_status(existing->edit_scene_graph().upsert_node(
            SceneNodeRecord{
                .uuid = fixed_uuid(9800),
                .type = "group",
                .name = "Original",
                .child_order = 0,
            }));
        ASSERT_TRUE(existing->save(path, save_options(1980, 1000)));

        auto opened = require_result_ptr(ProjectDocument::open(path));
        const auto generation_before = opened->generation();
        ASSERT_EQ(opened->project_uuid(), container_uuid);
        auto created_at = opened->project().created_at_unix_ns();
        ASSERT_TRUE(created_at);
        const auto original_created_at = *created_at;

        auto foreign = make_empty_document(foreign_uuid, 999);
        opened->edit_project() = foreign->project();
        ASSERT_TRUE(opened->edit_project().set_project_uuid(
            container_uuid));
        ASSERT_TRUE(opened->edit_project().set_created_at_unix_ns(
            original_created_at));

        opened->edit_scene_graph() = SceneGraphChapter();
        require_status(opened->edit_scene_graph().upsert_node(
            SceneNodeRecord{
                .uuid = adopted_node,
                .type = "splat",
                .name = "AdoptedHead",
                .child_order = 0,
                .payload = PayloadBinding{
                    .fourcc = "CKPT",
                    .instance_uuid = checkpoint_uuid,
                    .source_kind = "training",
                },
            }));
        require_status(
            opened->edit_scene_graph().set_training_model_uuid(
                adopted_node));
        for (const auto& uuid : opened->checkpoint_uuids()) {
            static_cast<void>(opened->remove_checkpoint(uuid));
        }
        lfs::core::CheckpointHeader header;
        header.iteration = 11;
        header.num_gaussians = 2;
        std::vector<std::byte> checkpoint_bytes(sizeof(header));
        std::memcpy(
            checkpoint_bytes.data(), &header, sizeof(header));
        auto payload = LazyChunkValue::from_owned(
            std::move(checkpoint_bytes), checkpoint_uuid);
        ASSERT_TRUE(payload)
            << lfs::format_for_developer(payload.error());
        require_status(opened->set_checkpoint(
            checkpoint_uuid, std::move(*payload)));

        auto options = save_options(1981, 1100);
        options.commit.snapshot_uuid = checkpoint_uuid;
        auto saved = opened->save(path, options);
        ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());

        auto reopened = require_result_ptr(ProjectDocument::open(path));
        EXPECT_EQ(reopened->project_uuid(), container_uuid);
        EXPECT_GT(reopened->generation(), generation_before);
        auto head = reopened->scene_graph().find(adopted_node);
        ASSERT_TRUE(head);
        ASSERT_TRUE(*head);
        EXPECT_EQ((*head)->name, "AdoptedHead");
        const auto bound = reopened->checkpoint_uuids();
        ASSERT_EQ(bound.size(), 1u);
        EXPECT_EQ(bound.front(), checkpoint_uuid);
        EXPECT_NE(reopened->find_checkpoint(checkpoint_uuid), nullptr);
    }

    TEST(ProjectDocumentTest,
         SaveAsToCurrentMasterPathRemainsSupported) {
        TemporaryDirectory temporary;
        const auto destination = temporary.path / "master.licht";
        auto document = make_empty_document(fixed_uuid(974), 100);
        ASSERT_TRUE(document->save(destination, save_options(1974, 1400)));
        auto opened = require_result_ptr(ProjectDocument::open(destination));

        ASSERT_TRUE(opened->save_as(destination, save_options(1975, 1500)));
        auto reader = ProjectReader::open(destination);
        ASSERT_TRUE(reader);
        EXPECT_EQ(reader->superblock().project_uuid, fixed_uuid(974));
    }

    TEST(ProjectDocumentPerformanceGate,
         MultiGigabyteCkptColdShellRestoreAndPartialSave) {
        const char* source_text = std::getenv("LFS_P6_P4_PROJECT_FIXTURE");
        const char* output_text = std::getenv("LFS_P6_LARGE_PROJECT_PATH");
        if (!source_text || !output_text) {
            GTEST_SKIP()
                << "Set LFS_P6_P4_PROJECT_FIXTURE and "
                   "LFS_P6_LARGE_PROJECT_PATH to run the P6 "
                   "multi-gigabyte shell-restore gate.";
        }

        const fs::path source(source_text);
        const fs::path output(output_text);
        const bool reuse_large_project = std::getenv("LFS_P6_REUSE_LARGE_PROJECT") != nullptr;
        ASSERT_TRUE(fs::is_regular_file(source));
        if (!reuse_large_project) {
            std::error_code filesystem_error;
            fs::create_directories(output.parent_path(), filesystem_error);
            ASSERT_FALSE(filesystem_error) << filesystem_error.message();
            fs::copy_file(source, output, fs::copy_options::overwrite_existing,
                          filesystem_error);
            ASSERT_FALSE(filesystem_error) << filesystem_error.message();
        } else {
            ASSERT_TRUE(fs::is_regular_file(output));
        }

        ChunkInfo source_checkpoint;
        std::vector<CleanProof> carry_proofs;
        {
            auto reader = ProjectReader::open(output);
            ASSERT_TRUE(reader) << lfs::format_for_developer(reader.error());
            const auto row = std::ranges::find_if(reader->chunks(), [](const ChunkInfo& candidate) {
                return candidate.key.fourcc == FOURCC_CKPT && candidate.row_kind == RowKind::Live;
            });
            ASSERT_NE(row, reader->chunks().end());
            ASSERT_EQ(row->compression, Compression::Stored);
            source_checkpoint = *row;
            for (const auto& candidate : reader->chunks()) {
                if (!candidate.is_live() || candidate.key == source_checkpoint.key) {
                    continue;
                }
                auto proof = reader->make_clean_proof(candidate, 1);
                ASSERT_TRUE(proof) << lfs::format_for_developer(proof.error());
                carry_proofs.push_back(std::move(*proof));
            }
        }

        constexpr std::uint64_t TARGET_CKPT_BYTES =
            2ull * 1024 * 1024 * 1024 + 256ull * 1024 * 1024;
        ASSERT_GT(source_checkpoint.stored_bytes, 0u);
        const std::uint64_t repeats =
            (TARGET_CKPT_BYTES + source_checkpoint.stored_bytes - 1) /
            source_checkpoint.stored_bytes;
        const std::uint64_t expanded_bytes = reuse_large_project
                                                 ? source_checkpoint.stored_bytes
                                                 : repeats * source_checkpoint.stored_bytes;
        ASSERT_GT(expanded_bytes, 2ull * 1024 * 1024 * 1024);

        double publish_ms = 0.0;
        if (!reuse_large_project) {
            const auto publish_started = std::chrono::steady_clock::now();
            auto writer = ProjectWriter::append(output, AppendOptions{
                                                            .compatibility = {},
                                                            .index_compression = IndexCompression::Zstd,
                                                            .disk_reserve_bytes = 0,
                                                            .boundary_observer = {},
                                                        });
            ASSERT_TRUE(writer) << lfs::format_for_developer(writer.error());
            require_status(writer->plan_commit(CommitOptions{
                .kind = CommitKind::Explicit,
                .commit_uuid = lfs::core::generate_uuid_v4(),
                .snapshot_uuid = source_checkpoint.key.instance_uuid,
                .wallclock_unix_ns = 1,
            }));
            require_status(writer->preflight(expanded_bytes));
            for (const auto& proof : carry_proofs) {
                require_status(writer->reuse_if_clean(proof, 1));
            }
            auto destination = writer->begin_chunk(source_checkpoint.key, ChunkWriteOptions{
                                                                              .chunk_version = source_checkpoint.chunk_version,
                                                                              .compression = Compression::Stored,
                                                                              .tensor_payload = true,
                                                                              .block_crcs = true,
                                                                              .expected_stream_bytes = expanded_bytes,
                                                                          });
            ASSERT_TRUE(destination) << lfs::format_for_developer(destination.error());

            std::ifstream source_stream(source, std::ios::binary);
            ASSERT_TRUE(source_stream);
            std::vector<char> buffer(8ull * 1024 * 1024);
            for (std::uint64_t repeat = 0; repeat < repeats; ++repeat) {
                source_stream.clear();
                source_stream.seekg(static_cast<std::streamoff>(source_checkpoint.payload_offset));
                ASSERT_TRUE(source_stream);
                std::uint64_t remaining = source_checkpoint.stored_bytes;
                while (remaining != 0) {
                    const auto requested = static_cast<std::streamsize>(
                        std::min<std::uint64_t>(remaining, buffer.size()));
                    source_stream.read(buffer.data(), requested);
                    ASSERT_EQ(source_stream.gcount(), requested);
                    (*destination)->write(buffer.data(), requested);
                    ASSERT_TRUE(**destination);
                    remaining -= static_cast<std::uint64_t>(requested);
                }
            }
            require_status(writer->end_chunk());
            require_status(writer->commit());
            publish_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - publish_started)
                             .count();
        }

        {
            auto reader = ProjectReader::open(output);
            ASSERT_TRUE(reader) << lfs::format_for_developer(reader.error());
            const auto* checkpoint = reader->find(source_checkpoint.key);
            ASSERT_NE(checkpoint, nullptr);
            EXPECT_EQ(checkpoint->stored_bytes, expanded_bytes);
            EXPECT_EQ(checkpoint->row_kind, RowKind::Live);
        }

        const auto drop_file_cache = [&output] {
#if defined(__linux__)
            const int file = ::open(output.c_str(), O_RDONLY);
            ASSERT_GE(file, 0);
            EXPECT_EQ(::posix_fadvise(file, 0, 0, POSIX_FADV_DONTNEED), 0);
            EXPECT_EQ(::close(file), 0);
#endif
        };

        std::vector<double> cold_shell_ms;
        std::vector<double> cold_open_ms;
        for (int sample = 0; sample < 5; ++sample) {
            drop_file_cache();
            const auto started = std::chrono::steady_clock::now();
            auto document = ProjectDocument::open(output, ProjectDocumentOpenOptions{
                                                              .reader = {},
                                                              .geometry = {},
                                                              .defer_geometry_payloads = true,
                                                          });
            ASSERT_TRUE(document) << lfs::format_for_developer(document.error());
            const auto elapsed_ms = [&] {
                return std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - started)
                    .count();
            };
            cold_open_ms.push_back(elapsed_ms());
            auto session = lfs::vis::project::prepareGuiSessionRestore({
                .gui_layout = document->gui_layout(),
                .editor = document->editor(),
                .view = document->view(),
                .sequencer = document->sequencer(),
                .metrics = document->metrics(),
            });
            ASSERT_TRUE(session) << lfs::format_for_developer(session.error());
            auto parameters = document->parameters().snapshot();
            ASSERT_TRUE(parameters) << lfs::format_for_developer(parameters.error());
            Scene shell_scene;
            auto shell = lfs::io::project::stage_scene_shell(
                document->scene_graph(), shell_scene);
            ASSERT_TRUE(shell) << lfs::format_for_developer(shell.error());
            cold_shell_ms.push_back(elapsed_ms());
        }
        std::ranges::sort(cold_shell_ms);
        std::ranges::sort(cold_open_ms);
        const double p50_shell_ms = cold_shell_ms[cold_shell_ms.size() / 2];
        const double max_shell_ms = cold_shell_ms.back();
        const auto join_samples = [](const std::vector<double>& samples) {
            std::ostringstream result;
            for (std::size_t index = 0; index < samples.size(); ++index) {
                if (index != 0) {
                    result << ',';
                }
                result << samples[index];
            }
            return result.str();
        };
        std::cout << "P6_SHELL_SAMPLES open_cold_ms=" << join_samples(cold_open_ms)
                  << " shell_cold_ms=" << join_samples(cold_shell_ms) << '\n';
        EXPECT_LT(max_shell_ms, 100.0);

        auto partial = ProjectDocument::open(output, ProjectDocumentOpenOptions{
                                                         .reader = {},
                                                         .geometry = {},
                                                         .defer_geometry_payloads = true,
                                                     });
        ASSERT_TRUE(partial) << lfs::format_for_developer(partial.error());
        partial->edit_metrics().accumulated_training_seconds += 1.0;
        const auto save_started = std::chrono::steady_clock::now();
        auto partial_save_options = save_options(1100, 600);
        partial_save_options.commit.commit_uuid = lfs::core::generate_uuid_v4();
        partial_save_options.commit.snapshot_uuid = source_checkpoint.key.instance_uuid;
        partial_save_options.commit.wallclock_unix_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
        auto saved = partial->save(output, partial_save_options);
        if (!saved) {
            for (const auto& suppressed : saved.error().suppressed()) {
                std::cerr << "P6_SAVE_SUPPRESSED\n"
                          << lfs::format_for_developer(suppressed);
            }
        }
        ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
        const double partial_save_ms = std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - save_started)
                                           .count();
        EXPECT_LT(partial_save_ms, 250.0);
        EXPECT_GT(saved->reused_chunks, 0u);

        std::cout << "P6_PERF ckpt_bytes=" << expanded_bytes
                  << " publish_ms=" << publish_ms
                  << " open_cold_ms=" << join_samples(cold_open_ms)
                  << " shell_cold_ms=" << join_samples(cold_shell_ms)
                  << " shell_p50_ms=" << p50_shell_ms
                  << " shell_max_ms=" << max_shell_ms
                  << " partial_save_ms=" << partial_save_ms << '\n';
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
        auto mesh = make_triangle_mesh();
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

        auto document = make_empty_document(project_uuid, 100);
        document->edit_scene_graph() = std::move(*scene_chapter);
        document->edit_selection() = std::move(*selection_chapter);
        require_status(document->set_splat(
            imported_uuid, std::move(*splat_payload)));
        require_status(document->set_point_cloud(
            point_uuid, PointCloudPayload(point_cloud)));
        require_status(
            document->set_mesh(mesh_uuid, MeshPayload(mesh)));
        auto checkpoint_model = make_splat(2);
        lfs::training::MCMC checkpoint_strategy(
            *checkpoint_model);
        lfs::core::param::TrainingParameters
            checkpoint_parameters;
        checkpoint_parameters.optimization =
            lfs::core::param::
                OptimizationParameters::
                    mcmc_defaults();
        checkpoint_parameters.optimization.sh_degree = 0;
        checkpoint_parameters.optimization.max_cap = 2;
        checkpoint_strategy.initialize(
            checkpoint_parameters.optimization);
        std::ostringstream checkpoint_stream(
            std::ios::binary | std::ios::out);
        const auto serialized_checkpoint =
            lfs::training::serialize_checkpoint(
                checkpoint_stream, 17,
                checkpoint_strategy,
                checkpoint_parameters,
                nullptr, nullptr, nullptr, nullptr);
        ASSERT_TRUE(serialized_checkpoint)
            << lfs::format_for_developer(
                   serialized_checkpoint.error());
        const auto checkpoint_string =
            checkpoint_stream.str();
        std::vector<std::byte> checkpoint_bytes(
            checkpoint_string.size());
        std::memcpy(
            checkpoint_bytes.data(),
            checkpoint_string.data(),
            checkpoint_string.size());
        auto checkpoint_payload =
            LazyChunkValue::from_owned(
                std::make_shared<
                    const std::vector<std::byte>>(
                    std::move(checkpoint_bytes)),
                checkpoint_uuid);
        ASSERT_TRUE(checkpoint_payload)
            << lfs::format_for_developer(
                   checkpoint_payload.error());
        require_status(document->set_checkpoint(
            checkpoint_uuid,
            std::move(*checkpoint_payload)));

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

        auto pending_parameters =
            require_result(document->parameters().snapshot());
        pending_parameters.active_strategy = "mcmc";
        pending_parameters.mcmc_session.iterations = 5;
        pending_parameters.mcmc_current.iterations = 5;
        require_status(document->edit_parameters().set_snapshot(
            pending_parameters));

        auto duplicate_training_payload = SplatChapterPayload::capture(
            *checkpoint_model, SplatSourceKind::ImportedPly, false);
        ASSERT_TRUE(duplicate_training_payload)
            << lfs::format_for_developer(
                   duplicate_training_payload.error());
        require_status(document->set_splat(
            training_uuid, std::move(*duplicate_training_payload)));
        require_status(project.upsert_embed_decision(EmbedDecision{
            .uuid = training_uuid,
            .node_uuid = training_uuid,
            .payload_fourcc = "SPLT",
            .decision = "embedded",
            .reason = "dual-authority validation case",
        }));
        require_status(project.upsert_embedded_payload_provenance(
            provenance(training_uuid, "SPLT",
                       "assets/dual-authority.ply", 9)));
        auto dual_authority_options = save_options(90, 190);
        dual_authority_options.commit.snapshot_uuid = checkpoint_uuid;
        auto dual_authority =
            document->save(path, dual_authority_options);
        ASSERT_FALSE(dual_authority);
        EXPECT_EQ(dual_authority.error().code(),
                  lfs::ErrorCode::DataLoss)
            << lfs::format_for_developer(dual_authority.error());
        EXPECT_FALSE(fs::exists(path));
        EXPECT_TRUE(document->remove_splat(training_uuid));
        auto removed_decision = project.dom().array_remove(
            "embed_decisions", training_uuid.to_string());
        ASSERT_TRUE(removed_decision && *removed_decision);
        auto removed_provenance = project.dom().array_remove(
            "embedded_payloads", training_uuid.to_string());
        ASSERT_TRUE(removed_provenance && *removed_provenance);

        auto first_options = save_options(100, 200);
        first_options.commit.snapshot_uuid =
            checkpoint_uuid;
        auto first =
            document->save(path, first_options);
        ASSERT_TRUE(first) << lfs::format_for_developer(first.error());
        EXPECT_EQ(first->generation, 1u);
        EXPECT_EQ(first->rewritten_chunks, 14u);

        auto reopened = require_result_ptr(ProjectDocument::open(path));
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
        ASSERT_TRUE(hydrated.checkpoint_header);
        EXPECT_EQ(hydrated.checkpoint_header->iteration, 17);
        EXPECT_EQ(hydrated.pending_parameters.active_strategy, "mcmc");
        EXPECT_EQ(hydrated.pending_parameters.mcmc_current.iterations, 5u);
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

        auto second_options = save_options(110, 300);
        second_options.commit.snapshot_uuid =
            checkpoint_uuid;
        auto second =
            reopened->save(path, second_options);
        ASSERT_TRUE(second)
            << lfs::format_for_developer(second.error());
        EXPECT_EQ(second->generation, 2u);
        EXPECT_EQ(second->rewritten_chunks, 1u);
        EXPECT_EQ(second->reused_chunks, 13u);

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

        auto document = make_empty_document(project_uuid, 1000);
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

        auto reopened = require_result_ptr(ProjectDocument::open(path));
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

        auto relink_document = require_result_ptr(ProjectDocument::open(path));
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

    TEST(ProjectDocumentTest,
         AutosaveIsCompleteBoundOverlayAndDoesNotMutateMasterOrDirtyState) {
        TemporaryDirectory temporary;
        const fs::path master =
            temporary.path / "recovery.licht";
        const fs::path sidecar =
            autosave_sidecar_path(master);
        const fs::path recovered =
            temporary.path / "recovered.licht";
        write_phase_a_fixture(master);

        const auto master_before =
            read_file_bytes(master);
        ProjectReader base =
            ProjectReader::open(master).value();
        auto document = require_result_ptr(ProjectDocument::open(master));
        require_status(
            document->edit_view().dom().set(
                "recovery_marker",
                std::string{"autosaved"}));
        const auto dirty_epoch =
            document->dirty_epoch();
        ASSERT_GT(dirty_epoch, 0u);

        const Uuid snapshot_uuid =
            fixed_uuid(980);
        auto saved = document->save_autosave(
            sidecar,
            ProjectDocumentAutosaveOptions{
                .file_uuid = fixed_uuid(981),
                .base_explicit_commit_uuid =
                    base.commit().commit_uuid,
                .autosave_sequence = 7,
                .snapshot_uuid = snapshot_uuid,
                .index_compression =
                    IndexCompression::
                        StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            });
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(saved.error());
        EXPECT_EQ(read_file_bytes(master),
                  master_before);
        EXPECT_TRUE(document->dirty());
        EXPECT_EQ(document->dirty_epoch(),
                  dirty_epoch);

        ProjectReader overlay =
            ProjectReader::open(sidecar).value();
        EXPECT_EQ(
            overlay.superblock().role,
            ContainerRole::AutosaveSidecar);
        EXPECT_EQ(
            overlay.superblock()
                .base_explicit_commit_uuid,
            base.commit().commit_uuid);
        EXPECT_EQ(
            overlay.superblock().autosave_sequence,
            7u);
        EXPECT_EQ(
            overlay.superblock()
                .sidecar_snapshot_uuid,
            snapshot_uuid);
        EXPECT_EQ(
            overlay.commit().kind,
            CommitKind::Autosave);
        require_status(overlay.verify_all());

        for (const auto& base_row : base.chunks()) {
            if (base_row.row_kind != RowKind::Live) {
                continue;
            }
            const auto found = std::ranges::find(
                overlay.chunks(), base_row.key,
                &ChunkInfo::key);
            ASSERT_NE(found, overlay.chunks().end())
                << base_row.key_string();
            EXPECT_TRUE(
                found->row_kind == RowKind::Live ||
                found->row_kind ==
                    RowKind::SidecarBaseReference ||
                found->row_kind ==
                    RowKind::Tombstone);
        }

        auto inspection =
            inspect_autosave_recovery(master);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(
                   inspection.error());
        EXPECT_EQ(
            inspection->disposition,
            RecoveryDisposition::Offer);
        EXPECT_EQ(
            inspection->selected_path,
            std::optional<fs::path>{sidecar});
        require_status(materialize_recovered_project(
            master, sidecar, recovered));
        ProjectReader materialized =
            ProjectReader::open(recovered).value();
        require_status(materialized.verify_all());
        const auto* overlay_view =
            overlay.find(FOURCC_VIEW,
                         base.superblock()
                             .project_uuid);
        const auto* recovered_view =
            materialized.find(
                FOURCC_VIEW,
                base.superblock().project_uuid);
        ASSERT_NE(overlay_view, nullptr);
        ASSERT_NE(recovered_view, nullptr);
        const auto overlay_view_bytes =
            overlay.read_chunk(*overlay_view);
        const auto recovered_view_bytes =
            materialized.read_chunk(
                *recovered_view);
        ASSERT_TRUE(overlay_view_bytes);
        ASSERT_TRUE(recovered_view_bytes);
        EXPECT_EQ(
            *overlay_view_bytes,
            *recovered_view_bytes);

        {
            auto held = ProjectWriter::append(
                master,
                AppendOptions{
                    .disk_reserve_bytes = 0,
                });
            ASSERT_TRUE(held)
                << lfs::format_for_developer(
                       held.error());
            auto locked_save =
                document->save_autosave(
                    sidecar,
                    ProjectDocumentAutosaveOptions{
                        .file_uuid =
                            fixed_uuid(982),
                        .base_explicit_commit_uuid =
                            base.commit()
                                .commit_uuid,
                        .autosave_sequence = 8,
                        .snapshot_uuid =
                            fixed_uuid(983),
                        .index_compression =
                            IndexCompression::
                                StoredForDeterministicTests,
                        .disk_reserve_bytes = 0,
                    });
            ASSERT_FALSE(locked_save);
            EXPECT_EQ(
                locked_save.error().code(),
                lfs::ErrorCode::Unavailable);
        }

        auto explicit_save =
            document->save(
                master, save_options(984, 300));
        ASSERT_TRUE(explicit_save)
            << lfs::format_for_developer(
                   explicit_save.error());
        auto stale =
            inspect_autosave_recovery(master);
        ASSERT_TRUE(stale)
            << lfs::format_for_developer(
                   stale.error());
        EXPECT_EQ(
            stale->disposition,
            RecoveryDisposition::StaleDeleted);
        EXPECT_FALSE(fs::exists(sidecar));
    }

    LazyChunkValue make_autosave_checkpoint_payload(
        const Uuid& checkpoint_uuid) {
        auto model = make_splat(2);
        lfs::training::MCMC strategy(*model);
        lfs::core::param::TrainingParameters parameters;
        parameters.optimization =
            lfs::core::param::OptimizationParameters::
                mcmc_defaults();
        parameters.optimization.sh_degree = 0;
        parameters.optimization.max_cap = 2;
        strategy.initialize(parameters.optimization);
        std::ostringstream stream(
            std::ios::binary | std::ios::out);
        (void)require_result(
            lfs::training::serialize_checkpoint(
                stream, 11, strategy, parameters,
                nullptr, nullptr, nullptr, nullptr));
        const auto encoded = stream.str();
        std::vector<std::byte> bytes(encoded.size());
        std::memcpy(
            bytes.data(), encoded.data(), encoded.size());
        auto payload = LazyChunkValue::from_owned(
            std::move(bytes), checkpoint_uuid);
        return require_result(std::move(payload));
    }

    void install_bound_autosave_checkpoint(
        ProjectDocument& document,
        const Uuid& training_uuid,
        const Uuid& checkpoint_uuid) {
        const Uuid root_uuid = fixed_uuid(951);
        require_status(
            document.edit_scene_graph().upsert_node(
                SceneNodeRecord{
                    .uuid = training_uuid,
                    .type = "splat",
                    .name = "Training",
                    .parent_uuid = root_uuid,
                    .child_order = 3,
                    .payload =
                        PayloadBinding{
                            .fourcc = "CKPT",
                            .instance_uuid =
                                checkpoint_uuid,
                            .source_kind = "training",
                        },
                }));
        require_status(
            document.edit_scene_graph()
                .set_training_model_uuid(training_uuid));
        require_status(document.set_checkpoint(
            checkpoint_uuid,
            make_autosave_checkpoint_payload(
                checkpoint_uuid)));
    }

    TEST(ProjectDocumentTest,
         AutosaveRejectsMismatchedSnapshotWhenCheckpointPresent) {
        TemporaryDirectory temporary;
        const fs::path master =
            temporary.path / "ckpt-mismatch.licht";
        const fs::path sidecar =
            autosave_sidecar_path(master);
        write_phase_a_fixture(master);
        ProjectReader base =
            require_result(ProjectReader::open(master));
        auto document =
            require_result_ptr(ProjectDocument::open(master));
        const Uuid checkpoint_uuid = fixed_uuid(9901);
        install_bound_autosave_checkpoint(
            *document, fixed_uuid(9900), checkpoint_uuid);
        require_status(
            document->edit_view().dom().set(
                "ckpt_mismatch_marker",
                std::string{"dirty"}));
        auto saved = document->save_autosave(
            sidecar,
            ProjectDocumentAutosaveOptions{
                .file_uuid = fixed_uuid(9902),
                .base_explicit_commit_uuid =
                    base.commit().commit_uuid,
                .autosave_sequence = 1,
                .snapshot_uuid = fixed_uuid(9903),
                .index_compression =
                    IndexCompression::
                        StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            });
        ASSERT_FALSE(saved);
        EXPECT_EQ(
            saved.error().code(),
            lfs::ErrorCode::FailedPrecondition);
        const auto formatted =
            lfs::format_for_developer(saved.error());
        EXPECT_NE(
            formatted.find("commit.snapshot_uuid"),
            std::string::npos)
            << formatted;
    }

    TEST(ProjectDocumentTest,
         AutosaveAdoptsCheckpointSnapshotWhenSnapshotUuidNil) {
        TemporaryDirectory temporary;
        const fs::path master =
            temporary.path / "ckpt-adopt.licht";
        const fs::path sidecar =
            autosave_sidecar_path(master);
        write_phase_a_fixture(master);
        ProjectReader base =
            require_result(ProjectReader::open(master));
        auto document =
            require_result_ptr(ProjectDocument::open(master));
        const Uuid checkpoint_uuid = fixed_uuid(9911);
        install_bound_autosave_checkpoint(
            *document, fixed_uuid(9910), checkpoint_uuid);
        require_status(
            document->edit_view().dom().set(
                "ckpt_adopt_marker",
                std::string{"dirty"}));
        auto saved = document->save_autosave(
            sidecar,
            ProjectDocumentAutosaveOptions{
                .file_uuid = fixed_uuid(9912),
                .base_explicit_commit_uuid =
                    base.commit().commit_uuid,
                .autosave_sequence = 1,
                .snapshot_uuid = {},
                .index_compression =
                    IndexCompression::
                        StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            });
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(saved.error());
        ProjectReader overlay =
            require_result(ProjectReader::open(sidecar));
        EXPECT_EQ(
            overlay.commit().snapshot_uuid,
            checkpoint_uuid);
        EXPECT_EQ(
            overlay.superblock().sidecar_snapshot_uuid,
            checkpoint_uuid);
        EXPECT_EQ(
            saved->snapshot_uuid, checkpoint_uuid);
    }

    TEST(ProjectDocumentTest,
         AutosaveMintsSnapshotWhenNoCheckpointAndSnapshotUuidNil) {
        TemporaryDirectory temporary;
        const fs::path master =
            temporary.path / "mint-snapshot.licht";
        const fs::path sidecar =
            autosave_sidecar_path(master);
        write_phase_a_fixture(master);
        ProjectReader base =
            require_result(ProjectReader::open(master));
        auto document =
            require_result_ptr(ProjectDocument::open(master));
        require_status(
            document->edit_view().dom().set(
                "mint_snapshot_marker",
                std::string{"dirty"}));
        auto saved = document->save_autosave(
            sidecar,
            ProjectDocumentAutosaveOptions{
                .file_uuid = fixed_uuid(9921),
                .base_explicit_commit_uuid =
                    base.commit().commit_uuid,
                .autosave_sequence = 1,
                .snapshot_uuid = {},
                .index_compression =
                    IndexCompression::
                        StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            });
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(saved.error());
        ProjectReader overlay =
            require_result(ProjectReader::open(sidecar));
        EXPECT_FALSE(
            overlay.commit().snapshot_uuid.is_nil());
        EXPECT_EQ(
            overlay.superblock().sidecar_snapshot_uuid,
            overlay.commit().snapshot_uuid);
        EXPECT_EQ(
            saved->snapshot_uuid,
            overlay.commit().snapshot_uuid);
    }

    TEST(ProjectDocumentTest,
         SaveAsDropsCheckpointRemovedBeforeRebind) {
        TemporaryDirectory temporary;
        const fs::path master =
            temporary.path / "ckpt-saveas-master.licht";
        const fs::path destination =
            temporary.path / "ckpt-saveas-copy.licht";
        write_phase_a_fixture(master);

        {
            auto document =
                require_result_ptr(ProjectDocument::open(master));
            install_bound_autosave_checkpoint(
                *document, fixed_uuid(9930), fixed_uuid(9931));
            auto options = save_options(9932, 300);
            options.commit.snapshot_uuid = fixed_uuid(9931);
            auto published = document->save(master, options);
            ASSERT_TRUE(published)
                << lfs::format_for_developer(
                       published.error());
        }

        auto document =
            require_result_ptr(ProjectDocument::open(master));
        const auto existing = document->checkpoint_uuids();
        ASSERT_EQ(existing.size(), 1u);
        EXPECT_EQ(existing.front(), fixed_uuid(9931));
        for (const auto& uuid : existing) {
            EXPECT_TRUE(document->remove_checkpoint(uuid));
        }
        install_bound_autosave_checkpoint(
            *document, fixed_uuid(9930), fixed_uuid(9933));

        auto options = save_options(9934, 400);
        options.commit.snapshot_uuid = fixed_uuid(9933);
        auto saved = document->save_as(destination, options);
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(saved.error());

        const auto remaining = document->checkpoint_uuids();
        ASSERT_EQ(remaining.size(), 1u);
        EXPECT_EQ(remaining.front(), fixed_uuid(9933));

        auto reader =
            require_result(ProjectReader::open(destination));
        std::vector<Uuid> live_ckpts;
        for (const auto& row : reader.chunks()) {
            if (!row.is_live()) {
                continue;
            }
            if (row.key.fourcc == FOURCC_CKPT) {
                live_ckpts.push_back(row.key.instance_uuid);
            }
        }
        ASSERT_EQ(live_ckpts.size(), 1u);
        EXPECT_EQ(live_ckpts.front(), fixed_uuid(9933));

        static_cast<void>(require_result_ptr(
            ProjectDocument::open(destination)));
    }

    TEST(ProjectDocumentTest,
         AutosaveInheritsMasterCompatibilityFloorsAndRequiredCapabilities) {
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "compatibility-master.licht";
        const fs::path sidecar = autosave_sidecar_path(master);
        write_phase_a_fixture(master);

        ProjectReader base = require_result(ProjectReader::open(master));
        std::vector<CleanProof> proofs;
        proofs.reserve(base.chunks().size());
        for (std::size_t index = 0; index < base.chunks().size(); ++index) {
            proofs.push_back(require_result(base.make_clean_proof(
                base.chunks()[index], 20'000 + index)));
        }
        {
            ProjectWriter writer = require_result(
                ProjectWriter::append(master, AppendOptions{
                                                  .compatibility = {},
                                                  .index_compression =
                                                      IndexCompression::StoredForDeterministicTests,
                                                  .disk_reserve_bytes = 0,
                                              }));
            CommitOptions elevated{
                .kind = CommitKind::Explicit,
                .commit_uuid = fixed_uuid(9863),
                .snapshot_uuid = fixed_uuid(9864),
                .wallclock_unix_ns = 1300,
                .min_reader_version = Version{1, 1},
                .min_safe_writer_version = Version{1, 1},
            };
            elevated.extra_reader_capabilities.set(100);
            elevated.extra_writer_capabilities.set(101);
            require_status(writer.plan_commit(elevated));
            require_status(writer.preflight(0));
            for (std::size_t index = 0; index < proofs.size(); ++index) {
                require_status(writer.reuse_if_clean(
                    proofs[index], 20'000 + index));
            }
            require_status(writer.commit());
        }

        ReaderOptions compatibility;
        compatibility.reader_version = Version{1, 1};
        compatibility.writer_version = Version{1, 1};
        compatibility.reader_capabilities.set(100);
        compatibility.writer_capabilities.set(101);
        ProjectReader elevated_master = require_result(
            ProjectReader::open(master, compatibility));
        ASSERT_TRUE(elevated_master.commit()
                        .required_reader_capabilities.contains(100));
        ASSERT_TRUE(elevated_master.commit()
                        .required_writer_capabilities.contains(101));
        auto document = require_result_ptr(ProjectDocument::open(
            master, ProjectDocumentOpenOptions{.reader = compatibility}));
        require_status(document->edit_view().dom().set(
            "compatibility_marker", std::string{"autosaved"}));
        auto saved = document->save_autosave(
            sidecar,
            ProjectDocumentAutosaveOptions{
                .file_uuid = fixed_uuid(9865),
                .base_explicit_commit_uuid =
                    elevated_master.commit().commit_uuid,
                .autosave_sequence = 1,
                .snapshot_uuid = fixed_uuid(9866),
                .index_compression =
                    IndexCompression::StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            });
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(saved.error());

        ProjectReader overlay = require_result(
            ProjectReader::open(sidecar, compatibility));
        EXPECT_TRUE(overlay.commit().min_reader_version >=
                    elevated_master.commit().min_reader_version);
        EXPECT_TRUE(overlay.commit().min_safe_writer_version >=
                    elevated_master.commit().min_safe_writer_version);
        EXPECT_TRUE(overlay.commit().required_reader_capabilities.contains_all(
            elevated_master.commit().required_reader_capabilities));
        EXPECT_TRUE(overlay.commit().required_writer_capabilities.contains_all(
            elevated_master.commit().required_writer_capabilities));
    }

    TEST(ProjectDocumentTest,
         RecoveredDocumentMergeHoldsMasterLockAndPublishesRecoveredKind) {
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "held-recovery.licht";
        const fs::path sidecar = autosave_sidecar_path(master);
        write_phase_a_fixture(master);
        ProjectReader base = require_result(ProjectReader::open(master));
        auto autosave_document = require_result_ptr(ProjectDocument::open(master));
        require_status(autosave_document->edit_view().dom().set(
            "held_recovery_marker", std::string{"recovered"}));
        auto autosaved = autosave_document->save_autosave(
            sidecar,
            ProjectDocumentAutosaveOptions{
                .file_uuid = fixed_uuid(9867),
                .base_explicit_commit_uuid = base.commit().commit_uuid,
                .autosave_sequence = 1,
                .snapshot_uuid = fixed_uuid(9868),
                .index_compression =
                    IndexCompression::StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            });
        ASSERT_TRUE(autosaved)
            << lfs::format_for_developer(autosaved.error());

        RecoverySession session =
            require_result(begin_recovery_session(master, sidecar));
        const fs::path staging = recovery_session_temp_path(master);
        require_status(materialize_recovered_project(
            master, sidecar, staging, session));
        auto recovered = require_result_ptr(ProjectDocument::open(staging));
        const auto recovered_marker =
            recovered->view().dom().get_json("held_recovery_marker");
        ASSERT_TRUE(recovered_marker.has_value());
        EXPECT_EQ(*recovered_marker, "recovered");
        auto competing = ProjectWriter::append(master);
        ASSERT_FALSE(competing);
        EXPECT_EQ(competing.error().code(), lfs::ErrorCode::Unavailable);

        // d1dac3cfa: WriterLockLease::release only drops the flock after
        // remaining copies are destroyed. Destroy the merge copy before
        // session.release() so inspect can acquire the master lock.
        {
            auto merge_options = save_options(9869, 1400);
            merge_options.commit.kind = CommitKind::Recovered;
            merge_options.writer_lock_lease = session.writer_lock();
            auto merged = recovered->save_as(master, merge_options);
            ASSERT_TRUE(merged)
                << lfs::format_for_developer(merged.error());
        }
        ProjectReader durable = require_result(ProjectReader::open(master));
        EXPECT_EQ(durable.commit().kind, CommitKind::Recovered);
        auto durable_document = require_result_ptr(ProjectDocument::open(master));
        const auto durable_marker =
            durable_document->view().dom().get_json("held_recovery_marker");
        ASSERT_TRUE(durable_marker.has_value());
        EXPECT_EQ(*durable_marker, "recovered");
        EXPECT_TRUE(fs::exists(staging));
        require_status(session.release());
        EXPECT_FALSE(fs::exists(staging));
        auto stale = inspect_autosave_recovery(master);
        ASSERT_TRUE(stale)
            << lfs::format_for_developer(stale.error());
        EXPECT_EQ(stale->disposition, RecoveryDisposition::StaleDeleted);
        EXPECT_FALSE(fs::exists(sidecar));
    }

    TEST(ProjectDocumentTest,
         HeadlessRecoveryRebindKeepsLazyPayloadReadableAfterDurableMerge) {
        TemporaryDirectory temporary;
        const fs::path master =
            temporary.path / "headless-recovery-merge.licht";
        const fs::path sidecar =
            autosave_sidecar_path(master);
        write_phase_a_fixture(master);
        ProjectReader base =
            require_result(ProjectReader::open(master));
        auto autosave_document = require_result_ptr(ProjectDocument::open(master));
        require_status(
            autosave_document->edit_view().dom().set(
                "headless_recovery_marker",
                std::string{"recovered"}));
        auto autosaved =
            autosave_document->save_autosave(
                sidecar,
                ProjectDocumentAutosaveOptions{
                    .file_uuid = fixed_uuid(9870),
                    .base_explicit_commit_uuid =
                        base.commit().commit_uuid,
                    .autosave_sequence = 1,
                    .snapshot_uuid = fixed_uuid(9871),
                    .index_compression =
                        IndexCompression::
                            StoredForDeterministicTests,
                    .disk_reserve_bytes = 0,
                });
        ASSERT_TRUE(autosaved)
            << lfs::format_for_developer(
                   autosaved.error());

        RecoverySession session = require_result(
            begin_recovery_session(master, sidecar));
        const fs::path staging =
            recovery_session_temp_path(master);
        require_status(materialize_recovered_project(
            master, sidecar, staging, session));
        auto recovered = require_result_ptr(ProjectDocument::open(
            staging,
            ProjectDocumentOpenOptions{
                .defer_geometry_payloads = true,
            }));
        ASSERT_TRUE(std::ranges::all_of(
            recovered->payload_states(),
            [](const auto& state) {
                return !state.loaded;
            }));

        lfs::app::detail::HeadlessRecoveryDocument owner(
            std::move(*recovered), std::move(session));
        ASSERT_NE(owner.recovery_session(), nullptr);
        EXPECT_TRUE(owner.recovery_session()
                        ->document_attached());
        RecoverySession trainer_session =
            *owner.recovery_session();

        // Mirror the trainer's durable recovered publish with a separate
        // document: the owner must remain bound to the staging file until it
        // explicitly reopens the new master head.
        auto writer_document = require_result_ptr(ProjectDocument::open(
            staging,
            ProjectDocumentOpenOptions{
                .defer_geometry_payloads = true,
            }));
        auto merge_options = save_options(9872, 1500);
        merge_options.commit.kind =
            CommitKind::Recovered;
        merge_options.writer_lock_lease =
            trainer_session.writer_lock();
        auto merged = writer_document->save_as(
            master, merge_options);
        ASSERT_TRUE(merged)
            << lfs::format_for_developer(
                   merged.error());
        ASSERT_TRUE(fs::exists(staging));

        auto early_release = trainer_session.release();
        ASSERT_FALSE(early_release);
        EXPECT_EQ(early_release.error().code(),
                  lfs::ErrorCode::FailedPrecondition);
        EXPECT_TRUE(fs::exists(staging));

        require_status(
            owner.rebind_after_durable_merge());
        ASSERT_TRUE(owner.document().source_path());
        EXPECT_EQ(owner.document().source_path()->lexically_normal(),
                  master.lexically_normal());
        EXPECT_FALSE(fs::exists(staging));

        // The deferred geometry was untouched before the durable merge. Its
        // first read now comes from the rebound master and must not see ENOENT.
        Scene hydrated_scene;
        auto shell = owner.document().stage_shell(hydrated_scene);
        ASSERT_TRUE(shell)
            << lfs::format_for_developer(
                   shell.error());
        hydrated_scene.commitRestoreStage(std::move(*shell));
        auto staged =
            owner.document().stage_hydration(hydrated_scene);
        ASSERT_TRUE(staged)
            << lfs::format_for_developer(
                   staged.error());
        const auto report =
            ProjectDocument::commit_partial_hydration(
                hydrated_scene, std::move(*staged), true);
        EXPECT_EQ(report.hydrated_payload_units, 3u);
        EXPECT_EQ(report.invalidated_payload_units, 0u);
    }

    TEST(ProjectDocumentTest,
         HeadlessRecoveryTeardownDeletesTempOnlyAfterDocumentDestruction) {
        TemporaryDirectory temporary;
        const fs::path master =
            temporary.path / "headless-recovery-teardown.licht";
        const fs::path sidecar =
            autosave_sidecar_path(master);
        write_phase_a_fixture(master);
        ProjectReader base =
            require_result(ProjectReader::open(master));
        auto autosave_document = require_result_ptr(ProjectDocument::open(master));
        require_status(
            autosave_document->edit_view().dom().set(
                "headless_teardown_marker",
                std::string{"recovered"}));
        auto autosaved =
            autosave_document->save_autosave(
                sidecar,
                ProjectDocumentAutosaveOptions{
                    .file_uuid = fixed_uuid(9873),
                    .base_explicit_commit_uuid =
                        base.commit().commit_uuid,
                    .autosave_sequence = 1,
                    .snapshot_uuid = fixed_uuid(9874),
                    .index_compression =
                        IndexCompression::
                            StoredForDeterministicTests,
                    .disk_reserve_bytes = 0,
                });
        ASSERT_TRUE(autosaved)
            << lfs::format_for_developer(
                   autosaved.error());

        const fs::path staging =
            recovery_session_temp_path(master);
        {
            RecoverySession session =
                require_result(begin_recovery_session(
                    master, sidecar));
            require_status(materialize_recovered_project(
                master, sidecar, staging, session));
            auto recovered = require_result_ptr(ProjectDocument::open(
                staging,
                ProjectDocumentOpenOptions{
                    .defer_geometry_payloads = true,
                }));
            lfs::app::detail::HeadlessRecoveryDocument owner(
                std::move(*recovered),
                std::move(session));
            ASSERT_TRUE(fs::exists(staging));
            RecoverySession trainer_session =
                *owner.recovery_session();
            auto refused = trainer_session.release();
            ASSERT_FALSE(refused);
            EXPECT_EQ(refused.error().code(),
                      lfs::ErrorCode::
                          FailedPrecondition);
            EXPECT_TRUE(fs::exists(staging));
        }
        EXPECT_FALSE(fs::exists(staging));
    }

    TEST(ProjectDocumentTest,
         TrainingAutosaveRecoveryPreservesInactiveParameterSlotsAndBindings) {
        TemporaryDirectory temporary;
        const fs::path master =
            temporary.path / "training-params.licht";
        const fs::path sidecar =
            autosave_sidecar_path(master);
        const fs::path recovered =
            temporary.path / "training-params-recovered.licht";
        const Uuid project_uuid = fixed_uuid(9850);
        const Uuid background_reference =
            fixed_uuid(9851);
        const Uuid ppisp_reference =
            fixed_uuid(9852);

        lfs::vis::ParameterManager manager;
        ASSERT_TRUE(manager.ensureLoaded());
        auto configured =
            manager.capturePendingProjectState();
        ASSERT_TRUE(configured)
            << lfs::format_for_developer(
                   configured.error());
        configured->active_strategy = "mrnf";
        configured->mcmc_session.iterations = 111;
        configured->mcmc_current.iterations = 112;
        configured->igs_session.iterations = 331;
        configured->igs_current.iterations = 332;
        configured->mcmc_session_references
            .background_image_reference =
            background_reference;
        configured->mcmc_current_references
            .background_image_reference =
            background_reference;
        configured->igs_session_references
            .ppisp_reference = ppisp_reference;
        configured->igs_current_references
            .ppisp_reference = ppisp_reference;
        require_status(
            manager.restorePendingProjectState(
                *configured));
        auto live_parameters =
            manager.capturePendingProjectState();
        ASSERT_TRUE(live_parameters)
            << lfs::format_for_developer(
                   live_parameters.error());

        Scene scene;
        const auto model_id = scene.addSplat(
            "training", make_splat(8));
        ASSERT_NE(model_id, lfs::core::NULL_NODE);
        scene.setTrainingModelNode(model_id);
        const auto model_uuid =
            scene.getNodeUuid(model_id);
        const std::array selected_nodes{model_uuid};
        lfs::training::ProjectSnapshotCpuState
            cpu_state;
        const auto snapshot_uuid = fixed_uuid(9853);
        auto captured = lfs::training::
            capture_project_snapshot_cpu_state(
                scene, *live_parameters,
                snapshot_uuid, 42, cpu_state,
                selected_nodes);
        ASSERT_TRUE(captured)
            << lfs::format_for_developer(
                   captured.error());
        lfs::training::ProjectSnapshotChapters
            chapters;
        require_status(lfs::training::
                           materialize_project_snapshot_cpu_chapters(
                               std::move(cpu_state),
                               chapters));
        EXPECT_EQ(
            chapters.selection
                .selected_node_uuids(),
            std::vector<Uuid>{model_uuid});

        auto document = make_empty_document(project_uuid, 1000);
        for (const auto& [uuid, key, kind] :
             std::array{
                 std::tuple{
                     background_reference,
                     std::string{"background"},
                     std::string{"image"}},
                 std::tuple{
                     ppisp_reference,
                     std::string{"ppisp"},
                     std::string{"ppisp"}},
             }) {
            require_status(
                document->edit_references().upsert(
                    ReferenceRecord{
                        .uuid = uuid,
                        .key = key,
                        .kind = kind,
                        .locator =
                            {
                                .preferred =
                                    "missing/" + key,
                                .base =
                                    LocatorBase::Project,
                            },
                        .fingerprint =
                            fake_fingerprint(42),
                        .unresolved = true,
                    }));
        }
        auto initial = save_options(9854, 1100);
        ASSERT_TRUE(document->save(master, initial));
        require_status(
            document->edit_parameters()
                .set_snapshot(
                    chapters.parameters));
        auto base = ProjectReader::open(master);
        ASSERT_TRUE(base);
        auto autosaved = document->save_autosave(
            sidecar,
            ProjectDocumentAutosaveOptions{
                .file_uuid = fixed_uuid(9857),
                .base_explicit_commit_uuid =
                    base->commit().commit_uuid,
                .autosave_sequence = 1,
                .snapshot_uuid = snapshot_uuid,
                .index_compression =
                    IndexCompression::
                        StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            });
        ASSERT_TRUE(autosaved)
            << lfs::format_for_developer(
                   autosaved.error());
        require_status(materialize_recovered_project(
            master, sidecar, recovered));
        auto reopened = require_result_ptr(ProjectDocument::open(recovered));
        auto restored =
            reopened->parameters().snapshot();
        ASSERT_TRUE(restored)
            << lfs::format_for_developer(
                   restored.error());
        EXPECT_EQ(restored->active_strategy,
                  "mrnf");
        EXPECT_EQ(restored->mcmc_session.iterations,
                  111u);
        EXPECT_EQ(restored->mcmc_current.iterations,
                  112u);
        EXPECT_EQ(restored->igs_session.iterations,
                  331u);
        EXPECT_EQ(restored->igs_current.iterations,
                  332u);
        EXPECT_EQ(
            restored->mcmc_session_references
                .background_image_reference,
            background_reference);
        EXPECT_EQ(
            restored->mcmc_current_references
                .background_image_reference,
            background_reference);
        EXPECT_EQ(
            restored->igs_session_references
                .ppisp_reference,
            ppisp_reference);
        EXPECT_EQ(
            restored->igs_current_references
                .ppisp_reference,
            ppisp_reference);
    }

    TEST(ProjectDocumentTest,
         TrainingAutosaveCarryForwardCkptRecoversNewestLightAndSpecifiedCkpt) {
        TemporaryDirectory temporary;
        const fs::path master =
            temporary.path / "train-light-ckpt.licht";
        const fs::path sidecar =
            autosave_sidecar_path(master);
        const fs::path recovered =
            temporary.path / "train-light-ckpt-recovered.licht";
        write_phase_a_fixture(master);
        auto seeded =
            require_result_ptr(ProjectDocument::open(master));
        const Uuid checkpoint_uuid = fixed_uuid(9931);
        install_bound_autosave_checkpoint(
            *seeded, fixed_uuid(9930), checkpoint_uuid);
        auto options = save_options(9932, 1400);
        options.commit.snapshot_uuid = checkpoint_uuid;
        auto published = seeded->save(master, options);
        ASSERT_TRUE(published)
            << lfs::format_for_developer(
                   published.error());

        auto document =
            require_result_ptr(ProjectDocument::open(master));
        const auto* clean =
            document->find_checkpoint(checkpoint_uuid);
        ASSERT_NE(clean, nullptr);
        EXPECT_TRUE(clean->is_clean_reference());
        require_status(
            document->edit_view().dom().set(
                "training_light_marker",
                std::string{"newest"}));
        auto base = require_result(ProjectReader::open(master));
        auto autosaved = document->save_autosave(
            sidecar,
            ProjectDocumentAutosaveOptions{
                .file_uuid = fixed_uuid(9933),
                .base_explicit_commit_uuid =
                    base.commit().commit_uuid,
                .autosave_sequence = 1,
                .snapshot_uuid = {},
                .index_compression =
                    IndexCompression::
                        StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            });
        ASSERT_TRUE(autosaved)
            << lfs::format_for_developer(
                   autosaved.error());

        ProjectReader overlay =
            require_result(ProjectReader::open(sidecar));
        const ChunkInfo* ckpt_row = overlay.find(
            FOURCC_CKPT, checkpoint_uuid);
        ASSERT_NE(ckpt_row, nullptr);
        EXPECT_EQ(
            ckpt_row->row_kind,
            RowKind::SidecarBaseReference);
        EXPECT_EQ(ckpt_row->payload_offset, 0u);
        for (const auto& row : overlay.chunks()) {
            if (row.key.fourcc == FOURCC_CKPT) {
                EXPECT_NE(
                    row.row_kind, RowKind::Live)
                    << row.key_string();
            }
        }

        auto offered = inspect_autosave_recovery(master);
        ASSERT_TRUE(offered)
            << lfs::format_for_developer(
                   offered.error());
        EXPECT_EQ(
            offered->disposition,
            RecoveryDisposition::Offer);

        require_status(materialize_recovered_project(
            master, sidecar, recovered));
        auto restored =
            require_result_ptr(ProjectDocument::open(recovered));
        const auto marker =
            restored->view().dom().get_json(
                "training_light_marker");
        ASSERT_TRUE(marker.has_value());
        EXPECT_EQ(*marker, "newest");
        const auto* recovered_ckpt =
            restored->find_checkpoint(checkpoint_uuid);
        ASSERT_NE(recovered_ckpt, nullptr);
        std::optional<int> iteration;
        auto visited = recovered_ckpt->visit_stream(
            [&](std::istream& stream,
                const std::uint64_t bytes)
                -> lfs::Result<void> {
                auto header =
                    lfs::core::load_checkpoint_header(
                        stream, bytes);
                if (header) {
                    iteration = header->iteration;
                }
                return {};
            });
        ASSERT_TRUE(visited)
            << lfs::format_for_developer(
                   visited.error());
        ASSERT_TRUE(iteration.has_value());
        EXPECT_EQ(*iteration, 11);
        auto training =
            restored->scene_graph().training_model_uuid();
        ASSERT_TRUE(training)
            << lfs::format_for_developer(
                   training.error());
        ASSERT_TRUE(training->has_value());
        EXPECT_EQ(**training, fixed_uuid(9930));
    }

    TEST(ProjectDocumentTest,
         OpenStreamsCkptWhenDecodedSizeExceedsMaterializeCap) {
        TemporaryDirectory temporary;
        const fs::path path =
            temporary.path / "ckpt-stream-cap.licht";
        const Uuid project_uuid = fixed_uuid(9940);
        const Uuid training_uuid = fixed_uuid(9941);
        const Uuid checkpoint_uuid = fixed_uuid(9942);
        auto document = make_empty_document(project_uuid, 100);
        require_status(document->edit_scene_graph().upsert_node(
            SceneNodeRecord{
                .uuid = training_uuid,
                .type = "splat",
                .name = "Training",
                .child_order = 0,
                .payload =
                    PayloadBinding{
                        .fourcc = "CKPT",
                        .instance_uuid = checkpoint_uuid,
                        .source_kind = "training",
                    },
            }));
        require_status(
            document->edit_scene_graph().set_training_model_uuid(
                training_uuid));
        require_status(document->set_checkpoint(
            checkpoint_uuid,
            make_autosave_checkpoint_payload(checkpoint_uuid)));
        auto options = save_options(9943, 1600);
        options.commit.snapshot_uuid = checkpoint_uuid;
        auto saved = document->save(path, options);
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(saved.error());

        {
            ProjectReader seed = require_result(ProjectReader::open(path));
            const ChunkInfo* ckpt =
                seed.find(FOURCC_CKPT, checkpoint_uuid);
            ASSERT_NE(ckpt, nullptr);
            auto ckpt_bytes = require_result(seed.read_chunk(*ckpt));
            ProjectWriter writer = require_result(
                ProjectWriter::append(
                    path,
                    AppendOptions{
                        .index_compression =
                            IndexCompression::
                                StoredForDeterministicTests,
                        .disk_reserve_bytes = 0,
                    }));
            require_status(writer.plan_commit(
                CommitOptions{
                    .kind = CommitKind::Explicit,
                    .commit_uuid = fixed_uuid(9944),
                    .snapshot_uuid = checkpoint_uuid,
                    .wallclock_unix_ns = 1700,
                }));
            require_status(writer.preflight(ckpt_bytes.size()));
            for (const auto& chunk : seed.chunks()) {
                if (!chunk.is_live()) {
                    continue;
                }
                if (chunk.key.fourcc == FOURCC_CKPT) {
                    require_status(writer.write_chunk(
                        chunk.key, ckpt_bytes,
                        ChunkWriteOptions{
                            .chunk_version = 1,
                            .compression = chunk.compression,
                            .tensor_payload = true,
                            .block_crcs = true,
                        }));
                    continue;
                }
                auto proof =
                    require_result(seed.make_clean_proof(chunk, 1));
                require_status(writer.reuse_if_clean(proof, 1));
            }
            require_status(writer.commit());
        }

        ProjectReader probe = require_result(ProjectReader::open(path));
        const ChunkInfo* row =
            probe.find(FOURCC_CKPT, checkpoint_uuid);
        ASSERT_NE(row, nullptr);
        ASSERT_GT(row->uncompressed_bytes, 1u);
        ASSERT_TRUE(row->block_crc_table.has_value());
        EXPECT_TRUE(row->compression == Compression::ZstdFramed ||
                    row->compression == Compression::ByteShuffleZstdFramed);

        struct PayloadCapGuard {
            explicit PayloadCapGuard(const std::uint64_t value) {
                detail::set_max_payload_materialized_bytes_for_testing(value);
            }
            PayloadCapGuard(const PayloadCapGuard&) = delete;
            PayloadCapGuard& operator=(const PayloadCapGuard&) = delete;
            ~PayloadCapGuard() {
                detail::set_max_payload_materialized_bytes_for_testing(
                    std::nullopt);
            }
        };
        const PayloadCapGuard cap(row->uncompressed_bytes - 1);
        auto materialized = probe.read_chunk(*row);
        ASSERT_FALSE(materialized);
        EXPECT_EQ(materialized.error().code(),
                  lfs::ErrorCode::ResourceExhausted);

        auto opened = require_result_ptr(ProjectDocument::open(path));
        const auto* checkpoint = opened->find_checkpoint(checkpoint_uuid);
        ASSERT_NE(checkpoint, nullptr);
        std::optional<lfs::core::CheckpointHeader> header;
        std::optional<lfs::core::SplatData> splat;
        auto visited = checkpoint->visit_stream(
            [&](std::istream& stream, const std::uint64_t bytes)
                -> lfs::Result<void> {
                auto loaded_header =
                    lfs::core::load_checkpoint_header(stream, bytes);
                if (!loaded_header) {
                    return {};
                }
                header = *loaded_header;
                stream.clear();
                stream.seekg(0);
                if (!stream) {
                    return {};
                }
                auto loaded_splat =
                    lfs::core::load_checkpoint_splat_data(stream, bytes);
                if (loaded_splat) {
                    splat = std::move(*loaded_splat);
                }
                return {};
            });
        ASSERT_TRUE(visited)
            << lfs::format_for_developer(visited.error());
        ASSERT_TRUE(header.has_value());
        EXPECT_EQ(header->iteration, 11);
        EXPECT_EQ(header->num_gaussians, 2u);
        ASSERT_TRUE(splat.has_value());
        EXPECT_EQ(splat->size(), 2);
        EXPECT_EQ(splat->get_max_sh_degree(), 0);
        const auto expected = make_splat(2);
        EXPECT_EQ(tensor_bytes(splat->means_raw()),
                  tensor_bytes(expected->means_raw()));
    }

    TEST(ProjectDocumentTest,
         StreamHydrateDeferredSpltMatchesMaterializedAndContentHash) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "splat-stream.licht";
        const Uuid project_uuid = fixed_uuid(12100);
        const Uuid splat_uuid = fixed_uuid(12101);
        auto model = make_splat(8);
        model->set_frozen_ranges({{1, 2}, {5, 1}});
        write_one_splat_document(path, *model, project_uuid, splat_uuid, 12110);

        ProjectReader probe = require_result(ProjectReader::open(path));
        const ChunkInfo* row = probe.find(FOURCC_SPLT, splat_uuid);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->compression, Compression::ZstdFramed);
        auto raw_bytes = require_result(probe.read_chunk(*row));
        const auto raw_hash = xxh3_128(raw_bytes);

        auto materialized = require_result_ptr(ProjectDocument::open(path));
        ASSERT_NE(materialized->find_splat(splat_uuid), nullptr);
        Scene materialized_scene;
        auto materialized_report = materialized->hydrate(materialized_scene);
        ASSERT_TRUE(materialized_report)
            << lfs::format_for_developer(materialized_report.error());
        const auto* materialized_node =
            materialized_scene.getNodeByUuid(splat_uuid);
        ASSERT_NE(materialized_node, nullptr);
        ASSERT_NE(materialized_node->model, nullptr);

        auto deferred = require_result_ptr(ProjectDocument::open(
            path, ProjectDocumentOpenOptions{.defer_geometry_payloads = true}));
        EXPECT_EQ(deferred->find_splat(splat_uuid), nullptr);
        ASSERT_EQ(deferred->payload_states().size(), 1u);
        EXPECT_FALSE(deferred->payload_states().front().loaded);
        EXPECT_EQ(splat_provenance_hash(*deferred, splat_uuid), raw_hash);
        Scene deferred_scene;
        auto deferred_report = deferred->hydrate(deferred_scene);
        ASSERT_TRUE(deferred_report)
            << lfs::format_for_developer(deferred_report.error());
        const auto* deferred_node = deferred_scene.getNodeByUuid(splat_uuid);
        ASSERT_NE(deferred_node, nullptr);
        ASSERT_NE(deferred_node->model, nullptr);
        expect_splats_bit_equal(*materialized_node->model, *deferred_node->model);

        rewrite_splat_chunk(path, splat_uuid,
                            Compression::ByteShuffleZstdFramed, 12120);
        ProjectReader shuffled = require_result(ProjectReader::open(path));
        const ChunkInfo* shuffled_row = shuffled.find(FOURCC_SPLT, splat_uuid);
        ASSERT_NE(shuffled_row, nullptr);
        ASSERT_EQ(raw_bytes.size() % 4, 0u);
        EXPECT_EQ(shuffled_row->compression, Compression::ByteShuffleZstdFramed);
        auto shuffled_bytes = require_result(shuffled.read_chunk(*shuffled_row));
        EXPECT_EQ(shuffled_bytes, raw_bytes);

        auto shuffled_deferred = require_result_ptr(ProjectDocument::open(
            path, ProjectDocumentOpenOptions{.defer_geometry_payloads = true}));
        Scene shuffled_scene;
        auto shuffled_report = shuffled_deferred->hydrate(shuffled_scene);
        ASSERT_TRUE(shuffled_report)
            << lfs::format_for_developer(shuffled_report.error());
        const auto* shuffled_node = shuffled_scene.getNodeByUuid(splat_uuid);
        ASSERT_NE(shuffled_node, nullptr);
        ASSERT_NE(shuffled_node->model, nullptr);
        expect_splats_bit_equal(*materialized_node->model, *shuffled_node->model);
        EXPECT_EQ(splat_provenance_hash(*shuffled_deferred, splat_uuid),
                  xxh3_128(shuffled_bytes));
    }

    TEST(ProjectDocumentTest, StreamHydrateForwardsMonotonicPayloadProgress) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "splat-progress.licht";
        const Uuid project_uuid = fixed_uuid(12180);
        const Uuid splat_uuid = fixed_uuid(12181);
        auto model = make_splat(16);
        write_one_splat_document(path, *model, project_uuid, splat_uuid, 12190);

        struct WindowGuard {
            explicit WindowGuard(const std::size_t bytes) {
                detail::set_splat_stream_window_bytes_for_testing(bytes);
            }
            WindowGuard(const WindowGuard&) = delete;
            WindowGuard& operator=(const WindowGuard&) = delete;
            ~WindowGuard() {
                detail::set_splat_stream_window_bytes_for_testing(std::nullopt);
            }
        };
        const WindowGuard window(64);
        auto deferred = require_result_ptr(ProjectDocument::open(
            path, ProjectDocumentOpenOptions{.defer_geometry_payloads = true}));
        std::vector<std::pair<std::size_t, std::size_t>> reports;
        Scene live;
        auto staged = deferred->stage_hydration(
            live, {}, {},
            [&](const std::size_t consumed, const std::size_t total) {
                reports.emplace_back(consumed, total);
            });
        ASSERT_TRUE(staged) << lfs::format_for_developer(staged.error());
        ASSERT_GE(reports.size(), 2u);
        EXPECT_EQ(reports.back().first, reports.back().second);
        for (std::size_t i = 0; i < reports.size(); ++i) {
            EXPECT_EQ(reports[i].second, reports.back().second);
            if (i > 0) {
                EXPECT_LE(reports[i - 1].first, reports[i].first);
            }
        }
        EXPECT_GT(reports.back().first, 0u);
    }

    TEST(ProjectDocumentTest,
         OversizedSpltAutoDefersOnOpenAndStreamHydrates) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "splat-auto-defer.licht";
        const Uuid project_uuid = fixed_uuid(12130);
        const Uuid splat_uuid = fixed_uuid(12131);
        auto model = make_splat(6);
        model->set_frozen_ranges({{0, 2}});
        auto reference_payload = require_result(SplatChapterPayload::capture(
            *model, SplatSourceKind::ImportedPly, false));
        auto reference = require_result(reference_payload.hydrate());
        write_one_splat_document(path, *model, project_uuid, splat_uuid, 12140);
        rewrite_splat_chunk(path, splat_uuid, Compression::ByteShuffleZstdFramed,
                            12141, true);

        ProjectReader probe = require_result(ProjectReader::open(path));
        const ChunkInfo* row = probe.find(FOURCC_SPLT, splat_uuid);
        ASSERT_NE(row, nullptr);
        ASSERT_GT(row->uncompressed_bytes, 1u);
        const auto raw_bytes = require_result(probe.read_chunk(*row));

        const PayloadCapGuard cap(row->uncompressed_bytes - 1);
        auto exhausted = probe.read_chunk(*row);
        ASSERT_FALSE(exhausted);
        EXPECT_EQ(exhausted.error().code(), lfs::ErrorCode::ResourceExhausted);

        auto opened = require_result_ptr(ProjectDocument::open(path));
        EXPECT_EQ(opened->find_splat(splat_uuid), nullptr);
        ASSERT_EQ(opened->payload_states().size(), 1u);
        EXPECT_EQ(opened->payload_states().front().fourcc, FOURCC_SPLT);
        EXPECT_EQ(opened->payload_states().front().instance_uuid, splat_uuid);
        EXPECT_FALSE(opened->payload_states().front().loaded);
        EXPECT_EQ(splat_provenance_hash(*opened, splat_uuid), xxh3_128(raw_bytes));

        Scene live;
        auto report = opened->hydrate(live);
        ASSERT_TRUE(report) << lfs::format_for_developer(report.error());
        const auto* node = live.getNodeByUuid(splat_uuid);
        ASSERT_NE(node, nullptr);
        ASSERT_NE(node->model, nullptr);
        expect_splats_bit_equal(*reference, *node->model);
    }

    TEST(ProjectDocumentTest,
         ByteShuffleZstdFallsBackWhenSpltSizeIsNotMultipleOfFour) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "splat-zstd-fallback.licht";
        const Uuid project_uuid = fixed_uuid(12150);
        const Uuid splat_uuid = fixed_uuid(12151);
        auto model = make_splat(3);
        model->deleted() = Tensor::zeros({std::size_t{3}}, Device::CPU,
                                         DataType::Bool);
        auto reference_payload = require_result(SplatChapterPayload::capture(
            *model, SplatSourceKind::ImportedPly, false));
        auto reference = require_result(reference_payload.hydrate());
        write_one_splat_document(path, *model, project_uuid, splat_uuid, 12160);

        ProjectReader before = require_result(ProjectReader::open(path));
        const ChunkInfo* source = before.find(FOURCC_SPLT, splat_uuid);
        ASSERT_NE(source, nullptr);
        auto raw_bytes = require_result(before.read_chunk(*source));
        ASSERT_NE(raw_bytes.size() % 4, 0u);

        rewrite_splat_chunk(path, splat_uuid,
                            Compression::ByteShuffleZstdFramed, 12170);
        ProjectReader after = require_result(ProjectReader::open(path));
        const ChunkInfo* row = after.find(FOURCC_SPLT, splat_uuid);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->compression, Compression::ZstdFramed);

        auto deferred = require_result_ptr(ProjectDocument::open(
            path, ProjectDocumentOpenOptions{.defer_geometry_payloads = true}));
        Scene live;
        auto report = deferred->hydrate(live);
        ASSERT_TRUE(report) << lfs::format_for_developer(report.error());
        const auto* node = live.getNodeByUuid(splat_uuid);
        ASSERT_NE(node, nullptr);
        ASSERT_NE(node->model, nullptr);
        expect_splats_bit_equal(*reference, *node->model);
        EXPECT_EQ(splat_provenance_hash(*deferred, splat_uuid),
                  xxh3_128(raw_bytes));
    }

    TEST(ProjectDocumentTest,
         SaveCopiesFileBackedCkptVerbatimWhenCleanProofIsLost) {
        TemporaryDirectory temporary;
        const fs::path path =
            temporary.path / "ckpt-verbatim-no-proof.licht";
        const Uuid project_uuid = fixed_uuid(9950);
        const Uuid training_uuid = fixed_uuid(9951);
        const Uuid checkpoint_uuid = fixed_uuid(9952);
        auto document = make_empty_document(project_uuid, 100);
        require_status(document->edit_scene_graph().upsert_node(
            SceneNodeRecord{
                .uuid = training_uuid,
                .type = "splat",
                .name = "Training",
                .child_order = 0,
                .payload =
                    PayloadBinding{
                        .fourcc = "CKPT",
                        .instance_uuid = checkpoint_uuid,
                        .source_kind = "training",
                    },
            }));
        require_status(
            document->edit_scene_graph().set_training_model_uuid(
                training_uuid));
        require_status(document->set_checkpoint(
            checkpoint_uuid,
            make_autosave_checkpoint_payload(checkpoint_uuid)));
        auto seed_options = save_options(9953, 1600);
        seed_options.commit.snapshot_uuid = checkpoint_uuid;
        auto seeded = document->save(path, seed_options);
        ASSERT_TRUE(seeded)
            << lfs::format_for_developer(seeded.error());

        std::vector<std::byte> logical_payload;
        {
            ProjectReader seed = require_result(ProjectReader::open(path));
            const ChunkInfo* ckpt =
                seed.find(FOURCC_CKPT, checkpoint_uuid);
            ASSERT_NE(ckpt, nullptr);
            logical_payload = require_result(seed.read_chunk(*ckpt));
            ProjectWriter writer = require_result(
                ProjectWriter::append(
                    path,
                    AppendOptions{
                        .index_compression =
                            IndexCompression::
                                StoredForDeterministicTests,
                        .disk_reserve_bytes = 0,
                    }));
            require_status(writer.plan_commit(
                CommitOptions{
                    .kind = CommitKind::Explicit,
                    .commit_uuid = fixed_uuid(9954),
                    .snapshot_uuid = checkpoint_uuid,
                    .wallclock_unix_ns = 1700,
                }));
            require_status(writer.preflight(logical_payload.size()));
            for (const auto& chunk : seed.chunks()) {
                if (!chunk.is_live()) {
                    continue;
                }
                if (chunk.key.fourcc == FOURCC_CKPT) {
                    require_status(writer.write_chunk(
                        chunk.key, logical_payload,
                        ChunkWriteOptions{
                            .chunk_version = 1,
                            .compression = chunk.compression,
                            .tensor_payload = true,
                            .block_crcs = true,
                        }));
                    continue;
                }
                auto proof =
                    require_result(seed.make_clean_proof(chunk, 1));
                require_status(writer.reuse_if_clean(proof, 1));
            }
            require_status(writer.commit());
        }

        ProjectReader probe = require_result(ProjectReader::open(path));
        const ChunkInfo* source_row =
            probe.find(FOURCC_CKPT, checkpoint_uuid);
        ASSERT_NE(source_row, nullptr);
        ASSERT_GT(source_row->uncompressed_bytes, 1u);
        ASSERT_TRUE(source_row->block_crc_table.has_value());
        EXPECT_TRUE(source_row->compression == Compression::ZstdFramed ||
                    source_row->compression ==
                        Compression::ByteShuffleZstdFramed);
        const auto source_compression = source_row->compression;
        const auto source_uncompressed = source_row->uncompressed_bytes;
        const auto source_version = source_row->chunk_version;
        const auto source_block_crc_count =
            source_row->block_crc_table->entries.size();
        const auto source_payload_offset = source_row->payload_offset;
        const auto source_generation = probe.commit().generation;

        auto opened = require_result_ptr(ProjectDocument::open(path));
        const auto* checkpoint = opened->find_checkpoint(checkpoint_uuid);
        ASSERT_NE(checkpoint, nullptr);
        EXPECT_TRUE(checkpoint->is_clean_reference());
        opened->drop_checkpoint_clean_proof_for_testing(checkpoint_uuid);
        EXPECT_FALSE(checkpoint->is_clean_reference());

        struct PayloadCapGuard {
            explicit PayloadCapGuard(const std::uint64_t value) {
                detail::set_max_payload_materialized_bytes_for_testing(value);
            }
            PayloadCapGuard(const PayloadCapGuard&) = delete;
            PayloadCapGuard& operator=(const PayloadCapGuard&) = delete;
            ~PayloadCapGuard() {
                detail::set_max_payload_materialized_bytes_for_testing(
                    std::nullopt);
            }
        };
        const PayloadCapGuard cap(source_uncompressed - 1);
        auto materialized = probe.read_chunk(*source_row);
        ASSERT_FALSE(materialized);
        EXPECT_EQ(materialized.error().code(),
                  lfs::ErrorCode::ResourceExhausted);

        auto rewrite_options = save_options(9955, 1800);
        rewrite_options.commit.snapshot_uuid = checkpoint_uuid;
        auto rewritten = opened->save(path, rewrite_options);
        ASSERT_TRUE(rewritten)
            << lfs::format_for_developer(rewritten.error());
        EXPECT_GE(rewritten->rewritten_chunks, 1u);
        EXPECT_GT(rewritten->generation, source_generation);

        auto reopened = require_result_ptr(ProjectDocument::open(path));
        const auto* restored = reopened->find_checkpoint(checkpoint_uuid);
        ASSERT_NE(restored, nullptr);

        std::vector<std::byte> restored_logical;
        std::optional<lfs::core::CheckpointHeader> header;
        bool restored_read_ok = false;
        auto visited = restored->visit_stream(
            [&](std::istream& stream, const std::uint64_t bytes)
                -> lfs::Result<void> {
                auto loaded_header =
                    lfs::core::load_checkpoint_header(stream, bytes);
                if (loaded_header) {
                    header = *loaded_header;
                }
                stream.clear();
                stream.seekg(0);
                restored_logical.resize(static_cast<std::size_t>(bytes));
                if (bytes != 0) {
                    stream.read(
                        reinterpret_cast<char*>(restored_logical.data()),
                        static_cast<std::streamsize>(bytes));
                }
                restored_read_ok =
                    static_cast<bool>(stream) &&
                    static_cast<std::uint64_t>(stream.gcount()) == bytes;
                return {};
            });
        ASSERT_TRUE(visited)
            << lfs::format_for_developer(visited.error());
        ASSERT_TRUE(header.has_value());
        EXPECT_EQ(header->iteration, 11);
        EXPECT_EQ(header->num_gaussians, 2u);
        ASSERT_TRUE(restored_read_ok);
        EXPECT_EQ(restored_logical, logical_payload);

        ProjectReader after = require_result(ProjectReader::open(path));
        const ChunkInfo* copied =
            after.find(FOURCC_CKPT, checkpoint_uuid);
        ASSERT_NE(copied, nullptr);
        EXPECT_EQ(copied->uncompressed_bytes, source_uncompressed);
        EXPECT_EQ(copied->compression, source_compression);
        EXPECT_EQ(copied->chunk_version, source_version);
        ASSERT_TRUE(copied->block_crc_table.has_value());
        EXPECT_EQ(copied->block_crc_table->entries.size(),
                  source_block_crc_count);
        EXPECT_NE(copied->payload_offset, source_payload_offset);
    }

    TEST(ProjectDocumentTest,
         HeadlessOpenWithoutRecoverKeepsSidecarUntilExplicitHeadAdvances) {
        TemporaryDirectory temporary;
        const fs::path master =
            temporary.path / "headless-decline.licht";
        const fs::path sidecar =
            autosave_sidecar_path(master);
        write_phase_a_fixture(master);
        auto base = ProjectReader::open(master);
        ASSERT_TRUE(base);
        auto autosave_document = require_result_ptr(ProjectDocument::open(master));
        require_status(
            autosave_document->edit_view().dom().set(
                "headless_recovery_marker",
                std::string{"autosaved"}));
        auto autosaved =
            autosave_document->save_autosave(
                sidecar,
                ProjectDocumentAutosaveOptions{
                    .file_uuid = fixed_uuid(9860),
                    .base_explicit_commit_uuid =
                        base->commit().commit_uuid,
                    .autosave_sequence = 9,
                    .snapshot_uuid =
                        fixed_uuid(9861),
                    .index_compression =
                        IndexCompression::
                            StoredForDeterministicTests,
                    .disk_reserve_bytes = 0,
                });
        ASSERT_TRUE(autosaved)
            << lfs::format_for_developer(
                   autosaved.error());
        auto offered =
            inspect_autosave_recovery(master);
        ASSERT_TRUE(offered);
        EXPECT_EQ(offered->disposition,
                  RecoveryDisposition::Offer);

        // This low-level document test opens the durable
        // master and leave the valid offer untouched.
        auto opened_master = require_result_ptr(ProjectDocument::open(master));
        EXPECT_TRUE(fs::is_regular_file(sidecar));
        EXPECT_FALSE(
            opened_master->view().dom().get_json(
                                           "headless_recovery_marker")
                .has_value());
        require_status(
            opened_master->edit_view().dom().set(
                "explicit_marker",
                std::string{"saved"}));
        auto options = save_options(9862, 1200);
        auto explicit_save =
            opened_master->save(master, options);
        ASSERT_TRUE(explicit_save)
            << lfs::format_for_developer(
                   explicit_save.error());
        EXPECT_TRUE(fs::is_regular_file(sidecar));
        auto stale =
            inspect_autosave_recovery(master);
        ASSERT_TRUE(stale);
        EXPECT_EQ(stale->disposition,
                  RecoveryDisposition::StaleDeleted);
        EXPECT_FALSE(fs::exists(sidecar));
    }

    TEST(ProjectDocumentTest,
         OpenInspectionDeletesOwnedOrphanCompactionTemps) {
        TemporaryDirectory temporary;
        const fs::path master =
            temporary.path / "stale-temp.licht";
        write_phase_a_fixture(master);
        const fs::path orphan =
            temporary.path /
            "stale-temp.compact.killed.1.0.tmp.licht";
        {
            std::ofstream output(
                orphan,
                std::ios::binary |
                    std::ios::trunc);
            output << "partial compaction";
        }
        ASSERT_TRUE(fs::exists(orphan));
        auto inspection =
            inspect_autosave_recovery(master);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(
                   inspection.error());
        EXPECT_EQ(
            inspection->disposition,
            RecoveryDisposition::StaleDeleted);
        EXPECT_FALSE(fs::exists(orphan));
        EXPECT_NE(
            std::ranges::find(
                inspection->deleted_paths,
                orphan),
            inspection->deleted_paths.end());
    }

#if defined(__linux__)
    TEST(ProjectDocumentTest,
         AutosaveSigkillAtEveryBoundaryLeavesCompleteOldOrNewSidecar) {
        TemporaryDirectory temporary;
        for (int boundary_value =
                 static_cast<int>(
                     CommitBoundary::
                         CurrentHeadValidated);
             boundary_value <=
             static_cast<int>(
                 CommitBoundary::Committed);
             ++boundary_value) {
            const auto boundary =
                static_cast<CommitBoundary>(
                    boundary_value);
            SCOPED_TRACE(std::format(
                "autosave boundary {}",
                boundary_value));
            const fs::path master =
                temporary.path /
                std::format(
                    "autosave-crash-{}.licht",
                    boundary_value);
            const fs::path sidecar =
                autosave_sidecar_path(master);
            write_phase_a_fixture(master);
            const auto master_before =
                read_file_bytes(master);
            const auto base =
                ProjectReader::open(master);
            ASSERT_TRUE(base);
            {
                auto document = require_result_ptr(ProjectDocument::open(master));
                require_status(
                    document->edit_view()
                        .dom()
                        .set(
                            "crash_marker",
                            std::string{"old"}));
                auto saved =
                    document->save_autosave(
                        sidecar,
                        ProjectDocumentAutosaveOptions{
                            .file_uuid =
                                fixed_uuid(
                                    1100 +
                                    boundary_value *
                                        10),
                            .base_explicit_commit_uuid =
                                base->commit()
                                    .commit_uuid,
                            .autosave_sequence =
                                1,
                            .snapshot_uuid =
                                fixed_uuid(
                                    1101 +
                                    boundary_value *
                                        10),
                            .index_compression =
                                IndexCompression::
                                    StoredForDeterministicTests,
                            .disk_reserve_bytes =
                                0,
                            .boundary_observer =
                                {},
                        });
                ASSERT_TRUE(saved)
                    << lfs::format_for_developer(
                           saved.error());
            }

            const pid_t child = ::fork();
            ASSERT_GE(child, 0);
            if (child == 0) {
                try {
                    auto document =
                        ProjectDocument::open(
                            master);
                    if (!document) {
                        ::_exit(101);
                    }
                    auto edited =
                        document->edit_view()
                            .dom()
                            .set(
                                "crash_marker",
                                std::string{"new"});
                    if (!edited) {
                        ::_exit(102);
                    }
                    auto saved =
                        document->save_autosave(
                            sidecar,
                            ProjectDocumentAutosaveOptions{
                                .file_uuid =
                                    fixed_uuid(
                                        1102 +
                                        boundary_value *
                                            10),
                                .base_explicit_commit_uuid =
                                    base->commit()
                                        .commit_uuid,
                                .autosave_sequence =
                                    2,
                                .snapshot_uuid =
                                    fixed_uuid(
                                        1103 +
                                        boundary_value *
                                            10),
                                .index_compression =
                                    IndexCompression::
                                        StoredForDeterministicTests,
                                .disk_reserve_bytes =
                                    0,
                                .boundary_observer =
                                    [boundary](
                                        const CommitBoundary
                                            reached) {
                                        if (reached ==
                                            boundary) {
                                            ::kill(
                                                ::getpid(),
                                                SIGKILL);
                                        }
                                    },
                            });
                    (void)saved;
                } catch (...) {
                    ::_exit(103);
                }
                ::_exit(104);
            }
            int status = 0;
            ASSERT_EQ(
                ::waitpid(child, &status, 0),
                child);
            ASSERT_TRUE(WIFSIGNALED(status));
            EXPECT_EQ(
                WTERMSIG(status), SIGKILL);
            EXPECT_EQ(
                read_file_bytes(master),
                master_before);

            auto inspection =
                inspect_autosave_recovery(
                    master);
            ASSERT_TRUE(inspection)
                << lfs::format_for_developer(
                       inspection.error());
            ASSERT_EQ(
                inspection->disposition,
                RecoveryDisposition::Offer);
            ASSERT_TRUE(
                inspection->selected_path);
            EXPECT_TRUE(
                inspection
                        ->autosave_sequence ==
                    1 ||
                inspection
                        ->autosave_sequence ==
                    2);
            const fs::path materialized =
                temporary.path /
                std::format(
                    "autosave-recovered-{}.licht",
                    boundary_value);
            require_status(
                materialize_recovered_project(
                    master,
                    *inspection
                         ->selected_path,
                    materialized));
            auto recovered =
                ProjectReader::open(
                    materialized);
            ASSERT_TRUE(recovered);
            require_status(
                recovered->verify_all());
            auto recovered_document =
                require_result_ptr(ProjectDocument::open(materialized));
            const auto payload_marker =
                recovered_document->view()
                    .dom()
                    .get<std::string>(
                        "crash_marker");
            ASSERT_TRUE(payload_marker);
            EXPECT_EQ(
                *payload_marker,
                inspection->autosave_sequence == 1
                    ? "old"
                    : "new");
        }
    }
#endif

} // namespace
