/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/project_container.hpp"
#include "io/project_document.hpp"
#include "io/project_recovery.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

    namespace fs = std::filesystem;
    using Json = nlohmann::ordered_json;
    using namespace lfs::io::project;

    constexpr std::uint64_t FIXED_CREATION_NS =
        1'735'689'600'000'000'000;
    constexpr std::uint64_t FIXED_COMMIT_NS =
        1'735'689'601'000'000'000;

    struct TemporaryDirectory {
        TemporaryDirectory() {
            static std::atomic_uint64_t counter{0};
            path = fs::temp_directory_path() /
                   std::format(
                       "lfs-p8-{}-{}",
                       std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count(),
                       counter.fetch_add(1));
            fs::create_directories(path);
        }

        ~TemporaryDirectory() {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }

        fs::path path;
    };

    lfs::core::Uuid fixed_uuid(const std::uint64_t tag) {
        const auto parsed = lfs::core::Uuid::from_string(
            std::format(
                "{:08x}-0000-4000-8000-{:012x}", tag,
                tag));
        if (!parsed) {
            std::abort();
        }
        return *parsed;
    }

    lfs::core::Uuid uuid_literal(const std::string_view text) {
        const auto parsed = lfs::core::Uuid::from_string(text);
        if (!parsed) {
            std::abort();
        }
        return *parsed;
    }

    ChunkKey fixed_key(const std::string_view fourcc,
                       const std::uint64_t tag) {
        const auto parsed = Fourcc::from_string(fourcc);
        if (!parsed) {
            std::abort();
        }
        return ChunkKey{
            .fourcc = *parsed,
            .instance_uuid = fixed_uuid(tag),
        };
    }

    template <typename T>
    T require_result(lfs::Result<T> result) {
        if (!result) {
            throw std::runtime_error(
                lfs::format_for_developer(result.error()));
        }
        return std::move(*result);
    }

    void require_status(lfs::Result<void> result) {
        if (!result) {
            throw std::runtime_error(
                lfs::format_for_developer(result.error()));
        }
    }

    std::vector<std::byte> bytes(const std::string_view text) {
        const auto view =
            std::as_bytes(std::span(text.data(), text.size()));
        return {view.begin(), view.end()};
    }

    std::vector<std::byte> read_file(const fs::path& path) {
        std::ifstream stream(path, std::ios::binary);
        const std::vector<char> raw{
            std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
        std::vector<std::byte> result(raw.size());
        if (!raw.empty()) {
            std::memcpy(result.data(), raw.data(), raw.size());
        }
        return result;
    }

    std::string sha256_file(const fs::path& path) {
        using Context =
            std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
        Context context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
        if (!context ||
            EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
            throw std::runtime_error("failed to initialize SHA-256");
        }
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error(
                std::format("failed to open {} for SHA-256", path.string()));
        }
        std::array<char, 1024 * 1024> buffer{};
        while (stream) {
            stream.read(buffer.data(),
                        static_cast<std::streamsize>(buffer.size()));
            const auto count = stream.gcount();
            if (count > 0 &&
                EVP_DigestUpdate(context.get(), buffer.data(),
                                 static_cast<std::size_t>(count)) != 1) {
                throw std::runtime_error("failed to update SHA-256");
            }
        }
        std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
        unsigned int digest_bytes = 0;
        if (EVP_DigestFinal_ex(context.get(), digest.data(),
                               &digest_bytes) != 1 ||
            digest_bytes != 32) {
            throw std::runtime_error("failed to finalize SHA-256");
        }
        std::string result;
        result.reserve(64);
        for (unsigned int index = 0; index < digest_bytes; ++index) {
            result += std::format("{:02x}",
                                  static_cast<unsigned int>(digest[index]));
        }
        return result;
    }

    ProjectDocumentSaveOptions save_options(
        const std::uint64_t tag,
        const CommitKind kind = CommitKind::Explicit,
        const IndexCompression compression =
            IndexCompression::Zstd) {
        return ProjectDocumentSaveOptions{
            .commit =
                {
                    .kind = kind,
                    .commit_uuid = fixed_uuid(tag),
                    .snapshot_uuid = fixed_uuid(tag + 1),
                    .wallclock_unix_ns = FIXED_COMMIT_NS + tag,
                },
            .file_uuid = fixed_uuid(tag + 2),
            .index_compression = compression,
            .disk_reserve_bytes = 0,
        };
    }

    std::vector<std::byte> tiny_png() {
        constexpr std::array<std::uint8_t, 67> data{
            0x89,
            0x50,
            0x4e,
            0x47,
            0x0d,
            0x0a,
            0x1a,
            0x0a,
            0x00,
            0x00,
            0x00,
            0x0d,
            0x49,
            0x48,
            0x44,
            0x52,
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
            0x01,
            0x08,
            0x06,
            0x00,
            0x00,
            0x00,
            0x1f,
            0x15,
            0xc4,
            0x89,
            0x00,
            0x00,
            0x00,
            0x0a,
            0x49,
            0x44,
            0x41,
            0x54,
            0x78,
            0x9c,
            0x63,
            0x60,
            0x00,
            0x00,
            0x00,
            0x02,
            0x00,
            0x01,
            0xe5,
            0x27,
            0xd4,
            0xa2,
            0x00,
            0x00,
            0x00,
            0x00,
            0x49,
            0x45,
            0x4e,
            0x44,
            0xae,
            0x42,
            0x60,
            0x82,
        };
        std::vector<std::byte> result(data.size());
        std::memcpy(result.data(), data.data(), data.size());
        return result;
    }

    ProjectDocument create_document(const std::uint64_t tag,
                                    const std::string_view producer) {
        auto document = require_result(ProjectDocument::create(
            fixed_uuid(tag), FIXED_CREATION_NS + tag));
        require_status(document.edit_project().dom().set(
            "p8_producer", std::string(producer)));
        return document;
    }

    void verify_open_fixture(const fs::path& path,
                             const CommitKind kind) {
        auto reader = ProjectReader::open(path);
        ASSERT_TRUE(reader)
            << lfs::format_for_developer(reader.error());
        EXPECT_EQ(reader->open_state(), OpenState::Open);
        EXPECT_EQ(reader->commit().kind, kind);
        EXPECT_LE(reader->commit().min_reader_version,
                  (Version{1, 0}));
        for (std::uint8_t bit = 8; bit < 128; ++bit) {
            EXPECT_FALSE(reader->commit()
                             .required_reader_capabilities
                             .contains(bit));
            EXPECT_FALSE(reader->commit()
                             .required_writer_capabilities
                             .contains(bit));
        }
        require_status(reader->verify_all());
    }

    void emit_release_corpus(const fs::path& output) {
        fs::create_directories(output);
        const fs::path saved = output / "save.licht";
        auto save_document = create_document(80'000, "save");
        require_status(save_document.edit_project().dom().set_json(
            "future_uuid_array",
            Json::array({fixed_uuid(80'010).to_string()})));
        auto saved_report =
            save_document.save(saved, save_options(80'100));
        ASSERT_TRUE(saved_report)
            << lfs::format_for_developer(saved_report.error());

        auto autosave_document = require_result(
            ProjectDocument::open(saved));
        require_status(autosave_document.edit_view().dom().set(
            "p8_autosave_marker", std::string("sidecar")));
        const auto base = require_result(ProjectReader::open(saved));
        const fs::path sidecar = autosave_sidecar_path(saved);
        auto autosaved = autosave_document.save_autosave(
            sidecar,
            ProjectDocumentAutosaveOptions{
                .file_uuid = fixed_uuid(80'200),
                .base_explicit_commit_uuid =
                    base.commit().commit_uuid,
                .autosave_sequence = 1,
                .snapshot_uuid = fixed_uuid(80'201),
                .commit_uuid = uuid_literal(
                    "54671195-b9fb-47b8-9e0a-006a4e6e2961"),
                .wallclock_unix_ns = 1'785'533'976'918'336'597,
                .index_compression = IndexCompression::Zstd,
                .disk_reserve_bytes = 0,
            });
        ASSERT_TRUE(autosaved)
            << lfs::format_for_developer(autosaved.error());

        const fs::path save_as = output / "save-as.licht";
        auto save_as_document = require_result(
            ProjectDocument::open(saved));
        require_status(save_as_document.edit_editor().dom().set(
            "p8_save_as_marker", std::string("save-as")));
        auto save_as_options = save_options(80'300);
        save_as_options.save_as_compaction_commit_uuid =
            uuid_literal(
                "144956bd-fa0a-4c74-948d-c4395dbfb187");
        save_as_options.save_as_creation_time_unix_ns =
            1'785'533'976'923'675'966;
        auto save_as_report = save_as_document.save_as(
            save_as, save_as_options);
        ASSERT_TRUE(save_as_report)
            << lfs::format_for_developer(save_as_report.error());

        const fs::path compacted = output / "compaction.licht";
        std::error_code copy_error;
        fs::copy_file(saved, compacted,
                      fs::copy_options::overwrite_existing,
                      copy_error);
        ASSERT_FALSE(copy_error) << copy_error.message();
        require_status(ProjectWriter::compact(
            compacted,
            CompactionOptions{
                .compatibility = {},
                .new_file_uuid = fixed_uuid(80'400),
                .commit_uuid = fixed_uuid(80'401),
                .snapshot_uuid = fixed_uuid(80'402),
                .creation_time_unix_ns = FIXED_CREATION_NS + 80'400,
                .wallclock_unix_ns = FIXED_COMMIT_NS + 80'400,
                .keep_tombstones = false,
                .disk_reserve_bytes = 0,
            }));

        const fs::path recovered = output / "recovered-commit.licht";
        require_status(materialize_recovered_project(
            saved,
            sidecar,
            recovered,
            {},
            RecoveryMaterializationOptions{
                .file_uuid = fixed_uuid(80'500),
                .commit_uuid = fixed_uuid(80'501),
                .wallclock_unix_ns = FIXED_COMMIT_NS + 80'500,
                .index_compression = IndexCompression::Zstd,
                .disk_reserve_bytes = 0,
            }));

        const fs::path preview = output / "foreign-preview.licht";
        auto preview_document =
            create_document(80'600, "foreign-preview");
        auto preview_options = save_options(80'610);
        const auto png = tiny_png();
        preview_options.preview_png = png;
        auto preview_report = preview_document.save(
            preview, preview_options);
        ASSERT_TRUE(preview_report)
            << lfs::format_for_developer(preview_report.error());

        verify_open_fixture(saved, CommitKind::Explicit);
        verify_open_fixture(sidecar, CommitKind::Autosave);
        verify_open_fixture(save_as, CommitKind::Explicit);
        verify_open_fixture(compacted, CommitKind::Compaction);
        verify_open_fixture(recovered, CommitKind::Recovered);
        verify_open_fixture(preview, CommitKind::Explicit);
        auto preview_reader = require_result(ProjectReader::open(preview));
        ASSERT_TRUE(preview_reader.preview().has_value());
        EXPECT_EQ(require_result(preview_reader.read_preview()), png);
    }

    ReaderOptions declared_v1() {
        ReaderOptions options;
        options.reader_version = Version{1, 0};
        options.writer_version = Version{1, 0};
        options.reader_capabilities = CapabilitySet{};
        options.writer_capabilities = CapabilitySet{};
        for (std::uint8_t bit = 0; bit < 8; ++bit) {
            options.reader_capabilities.set(bit);
            options.writer_capabilities.set(bit);
        }
        return options;
    }

    void create_raw_fixture(const fs::path& path,
                            const std::uint64_t tag,
                            const CommitOptions& commit) {
        auto writer = require_result(ProjectWriter::create(
            path,
            CreateOptions{
                .project_uuid = fixed_uuid(tag),
                .file_uuid = fixed_uuid(tag + 1),
                .role = ContainerRole::Master,
                .creation_time_unix_ns = FIXED_CREATION_NS + tag,
                .index_compression =
                    IndexCompression::StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            }));
        const auto payload = bytes("synthetic compatibility envelope");
        require_status(writer.plan_commit(commit));
        require_status(writer.preflight(payload.size()));
        require_status(writer.write_chunk(
            fixed_key("X1P8", tag + 2), payload));
        require_status(writer.commit());
    }

    void append_generation_with_requirements(
        const fs::path& path, const CommitOptions& commit,
        const std::optional<std::pair<ChunkKey,
                                      ChunkWriteOptions>>& replacement =
            std::nullopt,
        const std::span<const std::byte> replacement_payload = {},
        const IndexCompression index_compression =
            IndexCompression::Zstd) {
        auto prior = require_result(ProjectReader::open(path));
        std::vector<std::pair<ChunkInfo, CleanProof>> rows;
        for (std::size_t index = 0;
             index < prior.chunks().size(); ++index) {
            rows.emplace_back(
                prior.chunks()[index],
                require_result(prior.make_clean_proof(
                    prior.chunks()[index], 90'000 + index)));
        }
        auto writer = require_result(ProjectWriter::append(
            path,
            AppendOptions{
                .compatibility = {},
                .index_compression = index_compression,
                .disk_reserve_bytes = 0,
            }));
        require_status(writer.plan_commit(commit));
        require_status(writer.preflight(replacement_payload.size()));
        for (std::size_t index = 0; index < rows.size(); ++index) {
            if (replacement &&
                rows[index].first.key == replacement->first) {
                continue;
            }
            require_status(writer.reuse_if_clean(
                rows[index].second, 90'000 + index));
        }
        if (replacement) {
            require_status(writer.write_chunk(
                replacement->first, replacement_payload,
                replacement->second));
        }
        require_status(writer.commit());
    }

    struct StoredChunkWitness {
        ChunkInfo metadata;
        std::vector<std::byte> stored;
    };

    StoredChunkWitness witness(const ProjectReader& reader,
                               const ChunkKey& key) {
        const auto* row = reader.find(key);
        if (row == nullptr) {
            throw std::runtime_error("missing witness row");
        }
        std::vector<std::byte> stored(row->stored_bytes);
        require_status(reader.read_stored_at(*row, 0, stored));
        return StoredChunkWitness{*row, std::move(stored)};
    }

    void expect_same_opaque(const StoredChunkWitness& before,
                            const ProjectReader& after) {
        const auto* row = after.find(before.metadata.key);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->key, before.metadata.key);
        EXPECT_EQ(row->chunk_version,
                  before.metadata.chunk_version);
        EXPECT_EQ(row->compression, before.metadata.compression);
        EXPECT_EQ(row->flags, before.metadata.flags);
        EXPECT_EQ(row->stored_bytes, before.metadata.stored_bytes);
        EXPECT_EQ(row->uncompressed_bytes,
                  before.metadata.uncompressed_bytes);
        EXPECT_EQ(row->payload_crc32c,
                  before.metadata.payload_crc32c);
        EXPECT_EQ(row->block_crc_table.has_value(),
                  before.metadata.block_crc_table.has_value());
        std::vector<std::byte> stored(row->stored_bytes);
        require_status(after.read_stored_at(*row, 0, stored));
        EXPECT_EQ(stored, before.stored);
    }

    TEST(P8CompatibilityTest,
         ProductionReleaseCorpusEmissionPathsOpenAtFormatOneZero) {
        TemporaryDirectory temporary;
        const char* requested =
            std::getenv("LFS_P8_RELEASE_CORPUS_OUTPUT");
        const fs::path output =
            requested != nullptr && std::string_view(requested).size() != 0
                ? fs::path(requested)
                : temporary.path;
        emit_release_corpus(output);
    }

    TEST(P8CompatibilityTest,
         ReleaseCorpusProductionPathsReproduceManifestSha256) {
        TemporaryDirectory temporary;
        emit_release_corpus(temporary.path);

        const fs::path release =
            fs::path(PROJECT_ROOT_PATH) /
            "tools/licht_inspect/fixtures/release_corpus";
        std::ifstream manifest_stream(release / "manifest.json");
        ASSERT_TRUE(manifest_stream.good());
        Json manifest;
        manifest_stream >> manifest;

        std::size_t reproduced = 0;
        std::size_t documented_exemptions = 0;
        for (const auto& row : manifest.at("fixtures")) {
            const std::string relative = row.at("path");
            SCOPED_TRACE(relative);
            if (!row.at("reproducible").get<bool>()) {
                EXPECT_FALSE(row.value("reason", "").empty());
                ++documented_exemptions;
                continue;
            }
            const fs::path emitted = temporary.path / relative;
            ASSERT_TRUE(fs::is_regular_file(emitted));
            EXPECT_EQ(sha256_file(emitted),
                      row.at("sha256").get<std::string>());
            ++reproduced;
        }
        EXPECT_EQ(reproduced, 6u);
        EXPECT_EQ(documented_exemptions, 1u);

        const auto recovered = require_result(ProjectReader::open(
            temporary.path / "recovered-commit.licht"));
        EXPECT_EQ(recovered.commit().kind, CommitKind::Recovered);
        const auto& recovered_row = *std::ranges::find_if(
            manifest.at("fixtures"),
            [](const Json& row) {
                return row.at("path") == "recovered-commit.licht";
            });
        EXPECT_EQ(recovered_row.at("producer"),
                  "materialize_recovered_project");
    }

    TEST(P8CompatibilityTest,
         ReleaseCorpusCppReaderOpensEveryLockedArtifact) {
        const fs::path release =
            fs::path(PROJECT_ROOT_PATH) /
            "tools/licht_inspect/fixtures/release_corpus";
        constexpr std::array<std::string_view, 7> artifacts{
            "save.licht",
            "save.licht.autosave",
            "save-as.licht",
            "compaction.licht",
            "recovered-commit.licht",
            "headless-train-output.licht",
            "foreign-preview.licht",
        };
        const ReaderOptions v1 = declared_v1();
        for (const auto artifact : artifacts) {
            SCOPED_TRACE(artifact);
            const auto path = release / artifact;
            ASSERT_TRUE(fs::is_regular_file(path)) << path;
            const auto classification =
                ProjectReader::classify(path, v1);
            EXPECT_EQ(classification.state, OpenState::Open);
            auto reader = ProjectReader::open(path, v1);
            ASSERT_TRUE(reader)
                << lfs::format_for_developer(reader.error());
            EXPECT_LE(reader->commit().min_reader_version,
                      (Version{1, 0}));
            for (std::uint8_t bit = 8; bit < 128; ++bit) {
                EXPECT_FALSE(reader->commit()
                                 .required_reader_capabilities
                                 .contains(bit));
            }
            require_status(reader->verify_all());
        }

        auto preview = require_result(ProjectReader::open(
            release / "foreign-preview.licht", v1));
        ASSERT_TRUE(preview.preview().has_value());
        EXPECT_EQ(require_result(preview.read_preview()), tiny_png());
    }

    TEST(P8CompatibilityTest,
         V10CorpusAndSyntheticNewerClassificationRefusePayloadAccess) {
        TemporaryDirectory temporary;
        const auto supported = temporary.path / "supported.licht";
        auto supported_document = create_document(81'000, "supported");
        auto supported_save = supported_document.save(
            supported, save_options(81'010));
        ASSERT_TRUE(supported_save)
            << lfs::format_for_developer(supported_save.error());

        const ReaderOptions v1 = declared_v1();
        const auto classification =
            ProjectReader::classify(supported, v1);
        EXPECT_EQ(classification.state, OpenState::Open);

        for (const bool capability_gate : {false, true}) {
            const auto newer = temporary.path /
                               (capability_gate
                                    ? "newer-capability.licht"
                                    : "newer-version.licht");
            CommitOptions options{
                .kind = CommitKind::Explicit,
                .commit_uuid = fixed_uuid(
                    capability_gate ? 81'110 : 81'210),
                .snapshot_uuid = fixed_uuid(
                    capability_gate ? 81'111 : 81'211),
                .wallclock_unix_ns =
                    FIXED_COMMIT_NS +
                    (capability_gate ? 81'110 : 81'210),
            };
            if (capability_gate) {
                options.extra_reader_capabilities.set(8);
            } else {
                options.min_reader_version = Version{1, 1};
            }
            create_raw_fixture(
                newer, capability_gate ? 81'100 : 81'200,
                options);
            EXPECT_EQ(ProjectReader::classify(newer, v1).state,
                      OpenState::UnsupportedNewer);
            auto semantic = ProjectReader::open(newer, v1);
            ASSERT_FALSE(semantic);
            EXPECT_EQ(semantic.error().code(),
                      lfs::ErrorCode::Unsupported);

            auto inspect_options = v1;
            inspect_options.allow_unsupported_inspection = true;
            auto inspection = ProjectReader::open(
                newer, inspect_options);
            ASSERT_TRUE(inspection)
                << lfs::format_for_developer(inspection.error());
            EXPECT_EQ(inspection->open_state(),
                      OpenState::UnsupportedNewer);
            auto verified = inspection->verify_all();
            ASSERT_FALSE(verified);
            EXPECT_EQ(verified.error().code(),
                      lfs::ErrorCode::Unsupported);
            ASSERT_FALSE(inspection->chunks().empty());
            auto extracted = inspection->read_chunk(
                inspection->chunks().front());
            ASSERT_FALSE(extracted);
            EXPECT_EQ(extracted.error().code(),
                      lfs::ErrorCode::Unsupported);
            auto hydrated = ProjectDocument::open(
                newer,
                ProjectDocumentOpenOptions{.reader = v1});
            ASSERT_FALSE(hydrated);
            EXPECT_EQ(hydrated.error().code(),
                      lfs::ErrorCode::Unsupported);
        }

        const auto higher = temporary.path /
                            "unsupported-higher-head.licht";
        fs::copy_file(supported, higher);
        CommitOptions higher_commit{
            .kind = CommitKind::Explicit,
            .commit_uuid = fixed_uuid(81'300),
            .snapshot_uuid = fixed_uuid(81'301),
            .wallclock_unix_ns = FIXED_COMMIT_NS + 81'300,
        };
        higher_commit.extra_reader_capabilities.set(8);
        append_generation_with_requirements(
            higher, higher_commit, std::nullopt, {},
            IndexCompression::StoredForDeterministicTests);
        EXPECT_EQ(ProjectReader::classify(higher, v1).state,
                  OpenState::HardFail);
    }

    TEST(P8CompatibilityTest,
         ProhibitedOldWriterMatrixHasNoWriteEffectForEveryProducer) {
        enum class Gate { MinWriter,
                          Bit5,
                          Bit6,
                          Bit8,
                          VendorBit112 };
        constexpr std::array gates{
            Gate::MinWriter, Gate::Bit5, Gate::Bit6, Gate::Bit8,
            Gate::VendorBit112};
        TemporaryDirectory temporary;

        for (std::size_t index = 0; index < gates.size(); ++index) {
            const auto path = temporary.path /
                              std::format("gate-{}.licht", index);
            auto document = create_document(
                82'000 + index * 100,
                std::format("old-writer-gate-{}", index));
            auto options = save_options(
                82'010 + index * 100,
                CommitKind::Explicit,
                IndexCompression::StoredForDeterministicTests);
            if (gates[index] == Gate::MinWriter) {
                options.commit.min_safe_writer_version = Version{1, 1};
            } else if (gates[index] == Gate::Bit5) {
                options.commit.extra_writer_capabilities.set(5);
            } else if (gates[index] == Gate::Bit6) {
                options.commit.extra_writer_capabilities.set(6);
            } else if (gates[index] == Gate::Bit8) {
                options.commit.extra_writer_capabilities.set(8);
            } else {
                options.commit.extra_writer_capabilities.set(112);
            }
            auto initial = document.save(path, options);
            ASSERT_TRUE(initial)
                << lfs::format_for_developer(initial.error());

            ReaderOptions old = declared_v1();
            if (gates[index] == Gate::Bit5) {
                old.writer_capabilities.set(5, false);
            } else if (gates[index] == Gate::Bit6) {
                old.writer_capabilities.set(6, false);
            }
            const auto prior = require_result(
                ProjectReader::open(path, old));
            ASSERT_FALSE(prior.write_compatibility().safe);
            const auto before = read_file(path);
            const auto generation = prior.commit().generation;

            auto opened = ProjectDocument::open(
                path, ProjectDocumentOpenOptions{.reader = old});
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(opened.error());
            require_status(opened->edit_project().dom().set(
                "attempted_old_writer_edit", true));

            auto append_save = opened->save(
                path, save_options(82'500 + index * 10));
            ASSERT_FALSE(append_save);
            EXPECT_EQ(append_save.error().code(),
                      lfs::ErrorCode::Unsupported);
            EXPECT_EQ(read_file(path), before);

            const auto save_as = temporary.path /
                                 std::format("gate-{}-save-as.licht", index);
            auto save_as_result = opened->save_as(
                save_as, save_options(82'501 + index * 10));
            ASSERT_FALSE(save_as_result);
            EXPECT_EQ(save_as_result.error().code(),
                      lfs::ErrorCode::Unsupported);
            EXPECT_FALSE(fs::exists(save_as));
            EXPECT_EQ(read_file(path), before);

            const auto sidecar = autosave_sidecar_path(path);
            auto autosave = opened->save_autosave(
                sidecar,
                ProjectDocumentAutosaveOptions{
                    .file_uuid = fixed_uuid(82'502 + index * 10),
                    .base_explicit_commit_uuid =
                        prior.commit().commit_uuid,
                    .autosave_sequence = 1,
                    .snapshot_uuid = fixed_uuid(82'503 + index * 10),
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                });
            ASSERT_FALSE(autosave);
            EXPECT_EQ(autosave.error().code(),
                      lfs::ErrorCode::Unsupported);
            EXPECT_FALSE(fs::exists(sidecar));
            EXPECT_EQ(read_file(path), before);

            auto raw_append = ProjectWriter::append(
                path,
                AppendOptions{
                    .compatibility = old,
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                });
            ASSERT_FALSE(raw_append);
            EXPECT_EQ(raw_append.error().code(),
                      lfs::ErrorCode::Unsupported);
            EXPECT_EQ(read_file(path), before);

            auto compacted = ProjectWriter::compact(
                path,
                CompactionOptions{
                    .compatibility = old,
                    .new_file_uuid = fixed_uuid(82'504 + index * 10),
                    .commit_uuid = fixed_uuid(82'505 + index * 10),
                    .snapshot_uuid = fixed_uuid(82'506 + index * 10),
                    .creation_time_unix_ns = FIXED_CREATION_NS,
                    .wallclock_unix_ns = FIXED_COMMIT_NS,
                    .keep_tombstones = false,
                    .disk_reserve_bytes = 0,
                });
            ASSERT_FALSE(compacted);
            EXPECT_EQ(compacted.error().code(),
                      lfs::ErrorCode::Unsupported);
            EXPECT_EQ(read_file(path), before);
            EXPECT_EQ(require_result(ProjectReader::open(path, old))
                          .commit()
                          .generation,
                      generation);

            auto lease = WriterLockLease::acquire(path);
            ASSERT_TRUE(lease)
                << lfs::format_for_developer(lease.error());
        }
    }

    TEST(P8CompatibilityTest,
         OpaqueAndRetainedJsonSurviveSafeAppendAndCompaction) {
        TemporaryDirectory temporary;
        const auto path = temporary.path / "opaque-safe.licht";
        auto document = create_document(83'000, "opaque-safe");
        const Json future_uuids = Json::array(
            {fixed_uuid(83'001).to_string(),
             fixed_uuid(83'002).to_string()});
        require_status(document.edit_project().dom().set_json(
            "future_uuid_array", future_uuids));
        auto initial = document.save(path, save_options(83'010));
        ASSERT_TRUE(initial)
            << lfs::format_for_developer(initial.error());

        const auto base = require_result(ProjectReader::open(path));
        const ChunkKey view_key{
            .fourcc = FOURCC_VIEW,
            .instance_uuid = base.superblock().project_uuid,
        };
        const auto* base_view = base.find(view_key);
        ASSERT_NE(base_view, nullptr);
        const auto newer_view_payload =
            require_result(base.read_chunk(*base_view));

        CommitOptions opaque_commit{
            .kind = CommitKind::Explicit,
            .commit_uuid = fixed_uuid(83'020),
            .snapshot_uuid = fixed_uuid(83'021),
            .wallclock_unix_ns = FIXED_COMMIT_NS + 83'020,
        };
        opaque_commit.extra_writer_capabilities.set(
            OPAQUE_CHUNK_PRESERVATION);
        opaque_commit.extra_writer_capabilities.set(
            RETAINED_JSON_FIELDS);
        append_generation_with_requirements(
            path, opaque_commit,
            std::pair{
                view_key,
                ChunkWriteOptions{
                    .chunk_version = 99,
                    .compression = Compression::Zstd,
                }},
            newer_view_payload);

        const ChunkKey unknown_key = fixed_key("X8P8", 83'030);
        {
            auto prior = require_result(ProjectReader::open(path));
            std::vector<std::pair<ChunkInfo, CleanProof>> rows;
            for (std::size_t index = 0;
                 index < prior.chunks().size(); ++index) {
                rows.emplace_back(
                    prior.chunks()[index],
                    require_result(prior.make_clean_proof(
                        prior.chunks()[index], 91'000 + index)));
            }
            auto writer = require_result(ProjectWriter::append(
                path,
                AppendOptions{
                    .compatibility = {},
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                }));
            CommitOptions commit{
                .kind = CommitKind::Explicit,
                .commit_uuid = fixed_uuid(83'031),
                .snapshot_uuid = fixed_uuid(83'032),
                .wallclock_unix_ns = FIXED_COMMIT_NS + 83'031,
            };
            commit.extra_writer_capabilities.set(
                OPAQUE_CHUNK_PRESERVATION);
            commit.extra_writer_capabilities.set(
                RETAINED_JSON_FIELDS);
            const auto unknown_payload = bytes(
                "opaque bytes from a future X8P8 producer");
            require_status(writer.plan_commit(commit));
            require_status(writer.preflight(unknown_payload.size()));
            for (std::size_t index = 0; index < rows.size(); ++index) {
                require_status(writer.reuse_if_clean(
                    rows[index].second, 91'000 + index));
            }
            require_status(writer.write_chunk(
                unknown_key, unknown_payload,
                ChunkWriteOptions{
                    .chunk_version = 77,
                    .compression = Compression::Zstd,
                }));
            require_status(writer.commit());
        }

        auto before = require_result(ProjectReader::open(path));
        const auto unknown_before = witness(before, unknown_key);
        const auto view_before = witness(before, view_key);
        auto opened = ProjectDocument::open(path);
        ASSERT_TRUE(opened)
            << lfs::format_for_developer(opened.error());
        EXPECT_FALSE(opened->view().dom().get<std::string>(
                                             "p8_nonexistent")
                         .has_value());
        require_status(opened->edit_project().set_modified_at_unix_ns(
            FIXED_COMMIT_NS + 83'040));
        auto appended = opened->save(
            path, save_options(83'041));
        ASSERT_TRUE(appended)
            << lfs::format_for_developer(appended.error());
        EXPECT_EQ(appended->opaque_chunks_carried, 2u);

        auto after_append = require_result(ProjectReader::open(path));
        expect_same_opaque(unknown_before, after_append);
        expect_same_opaque(view_before, after_append);
        auto appended_document = require_result(
            ProjectDocument::open(path));
        EXPECT_EQ(appended_document.project().dom().get_json(
                      "future_uuid_array"),
                  std::optional<Json>(future_uuids));

        require_status(ProjectWriter::compact(
            path,
            CompactionOptions{
                .compatibility = {},
                .new_file_uuid = fixed_uuid(83'050),
                .commit_uuid = fixed_uuid(83'051),
                .snapshot_uuid = fixed_uuid(83'052),
                .creation_time_unix_ns = FIXED_CREATION_NS + 83'050,
                .wallclock_unix_ns = FIXED_COMMIT_NS + 83'050,
                .keep_tombstones = false,
                .disk_reserve_bytes = 0,
            }));
        auto after_compact = require_result(ProjectReader::open(path));
        expect_same_opaque(unknown_before, after_compact);
        expect_same_opaque(view_before, after_compact);
        auto compacted_document = require_result(
            ProjectDocument::open(path));
        EXPECT_EQ(compacted_document.project().dom().get_json(
                      "future_uuid_array"),
                  std::optional<Json>(future_uuids));
    }

    TEST(P8CompatibilityTest,
         MissingOpaqueOrRetainedJsonCapabilityMakesDocumentReadOnly) {
        TemporaryDirectory temporary;

        const auto opaque_path = temporary.path / "missing-bit5.licht";
        auto opaque_document = create_document(84'000, "missing-bit5");
        ASSERT_TRUE(opaque_document.save(
            opaque_path, save_options(84'010)));
        {
            auto prior = require_result(ProjectReader::open(opaque_path));
            std::vector<std::pair<ChunkInfo, CleanProof>> rows;
            for (std::size_t index = 0;
                 index < prior.chunks().size(); ++index) {
                rows.emplace_back(
                    prior.chunks()[index],
                    require_result(prior.make_clean_proof(
                        prior.chunks()[index], 92'000 + index)));
            }
            auto writer = require_result(ProjectWriter::append(
                opaque_path,
                AppendOptions{
                    .compatibility = {},
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                }));
            require_status(writer.plan_commit(CommitOptions{
                .kind = CommitKind::Explicit,
                .commit_uuid = fixed_uuid(84'020),
                .snapshot_uuid = fixed_uuid(84'021),
                .wallclock_unix_ns = FIXED_COMMIT_NS + 84'020,
            }));
            const auto payload = bytes("future opaque without declaration");
            require_status(writer.preflight(payload.size()));
            for (std::size_t index = 0; index < rows.size(); ++index) {
                require_status(writer.reuse_if_clean(
                    rows[index].second, 92'000 + index));
            }
            require_status(writer.write_chunk(
                fixed_key("X8N5", 84'022), payload));
            require_status(writer.commit());
        }
        auto opaque_open = require_result(
            ProjectDocument::open(opaque_path));
        require_status(opaque_open.edit_project().dom().set(
            "attempt", true));
        const auto opaque_before = read_file(opaque_path);
        auto opaque_save = opaque_open.save(
            opaque_path, save_options(84'030));
        ASSERT_FALSE(opaque_save);
        EXPECT_EQ(opaque_save.error().code(),
                  lfs::ErrorCode::Unsupported);
        EXPECT_EQ(read_file(opaque_path), opaque_before);

        const auto json_path = temporary.path / "missing-bit6.licht";
        auto json_document = require_result(ProjectDocument::create(
            fixed_uuid(84'100), FIXED_CREATION_NS + 84'100));
        ASSERT_TRUE(json_document.save(
            json_path, save_options(84'110)));
        auto prior = require_result(ProjectReader::open(json_path));
        const ChunkKey proj_key{
            .fourcc = FOURCC_PROJ,
            .instance_uuid = prior.superblock().project_uuid,
        };
        const auto* proj = prior.find(proj_key);
        ASSERT_NE(proj, nullptr);
        auto proj_payload = require_result(prior.read_chunk(*proj));
        Json proj_json = Json::parse(
            reinterpret_cast<const char*>(proj_payload.data()),
            reinterpret_cast<const char*>(proj_payload.data()) +
                proj_payload.size());
        proj_json["future_without_bit6"] = Json::array(
            {fixed_uuid(84'111).to_string()});
        const auto future_text = proj_json.dump(2);
        const auto future_payload = bytes(future_text);
        append_generation_with_requirements(
            json_path,
            CommitOptions{
                .kind = CommitKind::Explicit,
                .commit_uuid = fixed_uuid(84'120),
                .snapshot_uuid = fixed_uuid(84'121),
                .wallclock_unix_ns = FIXED_COMMIT_NS + 84'120,
            },
            std::pair{
                proj_key,
                ChunkWriteOptions{
                    .chunk_version = 1,
                    .compression = Compression::Zstd,
                }},
            future_payload);
        auto json_open = require_result(
            ProjectDocument::open(json_path));
        require_status(json_open.edit_project().set_modified_at_unix_ns(
            FIXED_COMMIT_NS + 84'130));
        const auto json_before = read_file(json_path);
        auto json_save = json_open.save(
            json_path, save_options(84'131));
        ASSERT_FALSE(json_save);
        EXPECT_EQ(json_save.error().code(),
                  lfs::ErrorCode::Unsupported);
        EXPECT_EQ(read_file(json_path), json_before);
    }

    TEST(P8CompatibilityTest,
         SidecarBaseReferenceAbsentFromMasterIsRejectedByRecoveryInspection) {
        TemporaryDirectory temporary;
        const auto master_path = temporary.path / "base-reference.licht";
        const auto sidecar_path = autosave_sidecar_path(master_path);
        auto document = create_document(84'500, "base-reference-master");
        ASSERT_TRUE(document.save(master_path, save_options(84'510)));
        auto master = require_result(ProjectReader::open(master_path));
        const auto snapshot_uuid = fixed_uuid(84'520);

        {
            auto writer = require_result(ProjectWriter::create(
                sidecar_path,
                CreateOptions{
                    .project_uuid = master.superblock().project_uuid,
                    .file_uuid = fixed_uuid(84'521),
                    .role = ContainerRole::AutosaveSidecar,
                    .base_explicit_commit_uuid = master.commit().commit_uuid,
                    .autosave_sequence = 1,
                    .sidecar_snapshot_uuid = snapshot_uuid,
                    .creation_time_unix_ns = FIXED_CREATION_NS + 84'521,
                    .index_compression =
                        IndexCompression::StoredForDeterministicTests,
                    .disk_reserve_bytes = 0,
                    .writer_lock_anchor = master_path,
                }));
            CommitOptions sidecar_commit{
                .kind = CommitKind::Autosave,
                .commit_uuid = fixed_uuid(84'522),
                .snapshot_uuid = snapshot_uuid,
                .wallclock_unix_ns = FIXED_COMMIT_NS + 84'522,
            };
            sidecar_commit.extra_reader_capabilities =
                master.commit().required_reader_capabilities;
            sidecar_commit.extra_writer_capabilities =
                master.commit().required_writer_capabilities;
            require_status(writer.plan_commit(sidecar_commit));
            require_status(writer.preflight(0));
            for (const auto& row : master.chunks()) {
                if (row.is_live()) {
                    require_status(writer.add_sidecar_base_reference(row));
                }
            }
            ASSERT_FALSE(master.chunks().empty());
            ChunkInfo absent = master.chunks().front();
            absent.key = fixed_key("XBAS", 84'523);
            require_status(writer.add_sidecar_base_reference(absent));
            require_status(writer.commit());
        }

        auto inspection = inspect_autosave_recovery(master_path);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(inspection.error());
        EXPECT_EQ(inspection->disposition, RecoveryDisposition::Invalid);
        ASSERT_FALSE(inspection->diagnostics.empty());
        EXPECT_TRUE(std::ranges::any_of(
            inspection->diagnostics,
            [](const std::string& diagnostic) {
                return diagnostic.find("invalid base reference") !=
                       std::string::npos;
            }));
    }

#ifndef _WIN32
    TEST(P8LockMatrixTest,
         SecondProcessIsReadOnlyOnOriginalAndMaySaveAsElsewhere) {
        TemporaryDirectory temporary;
        const auto original = temporary.path / "locked.licht";
        const auto save_as = temporary.path / "contender-save-as.licht";
        auto document = create_document(85'000, "lock-holder");
        ASSERT_TRUE(document.save(
            original, save_options(85'010)));
        const auto before = read_file(original);
        auto lease = require_result(
            WriterLockLease::acquire(original));

        const pid_t child = ::fork();
        ASSERT_GE(child, 0);
        if (child == 0) {
            const auto fail_child = [](const int code) {
                ::_exit(code);
            };
            auto reader = ProjectReader::open(original);
            if (!reader || reader->open_state() != OpenState::Open) {
                fail_child(10);
            }
            auto contender = ProjectDocument::open(original);
            if (!contender) {
                fail_child(11);
            }
            if (!contender->edit_project().dom().set(
                    "contender", true)) {
                fail_child(12);
            }
            auto blocked_save = contender->save(
                original, save_options(85'020));
            if (blocked_save ||
                blocked_save.error().code() !=
                    lfs::ErrorCode::Unavailable ||
                read_file(original) != before) {
                fail_child(13);
            }
            auto blocked_compact = ProjectWriter::compact(
                original,
                CompactionOptions{
                    .compatibility = {},
                    .new_file_uuid = fixed_uuid(85'021),
                    .commit_uuid = fixed_uuid(85'022),
                    .snapshot_uuid = fixed_uuid(85'023),
                    .creation_time_unix_ns = FIXED_CREATION_NS,
                    .wallclock_unix_ns = FIXED_COMMIT_NS,
                    .keep_tombstones = false,
                    .disk_reserve_bytes = 0,
                });
            if (blocked_compact ||
                blocked_compact.error().code() !=
                    lfs::ErrorCode::Unavailable ||
                read_file(original) != before) {
                fail_child(14);
            }
            const auto sidecar = autosave_sidecar_path(original);
            auto blocked_autosave = contender->save_autosave(
                sidecar,
                ProjectDocumentAutosaveOptions{
                    .file_uuid = fixed_uuid(85'024),
                    .base_explicit_commit_uuid =
                        reader->commit().commit_uuid,
                    .autosave_sequence = 1,
                    .snapshot_uuid = fixed_uuid(85'025),
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                });
            if (blocked_autosave ||
                blocked_autosave.error().code() !=
                    lfs::ErrorCode::Unavailable ||
                fs::exists(sidecar) ||
                read_file(original) != before) {
                fail_child(15);
            }
            auto allowed_save_as = contender->save_as(
                save_as, save_options(85'030));
            if (!allowed_save_as || !fs::is_regular_file(save_as)) {
                fail_child(16);
            }
            auto saved_reader = ProjectReader::open(save_as);
            if (!saved_reader || !saved_reader->verify_all()) {
                fail_child(17);
            }
            fail_child(0);
        }

        int status = 0;
        ASSERT_EQ(::waitpid(child, &status, 0), child);
        ASSERT_TRUE(WIFEXITED(status));
        EXPECT_EQ(WEXITSTATUS(status), 0);
        EXPECT_EQ(read_file(original), before);
        EXPECT_TRUE(fs::is_regular_file(save_as));
    }
#else
    TEST(P8LockMatrixTest,
         WindowsLockFileExCellIsExplicitlyDeferredToP0d) {
        GTEST_SKIP()
            << "Windows LockFileEx two-process matrix is OWED by P0d; "
               "this is an explicit platform deferral, never a success stub";
    }
#endif

} // namespace
