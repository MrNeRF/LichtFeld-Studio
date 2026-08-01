/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/project_document.hpp"
#include "licht_matrix_test_data.hpp"
#include "licht_test_support.hpp"
#include "ppisp_fixture.hpp"
#include "training/checkpoint.hpp"
#include "training/components/bilateral_grid.hpp"
#include "training/components/ppisp.hpp"
#include "training/components/ppisp_controller_pool.hpp"
#include "training/components/ppisp_file.hpp"
#include "training/components/sparsity_optimizer.hpp"
#include "training/strategies/mcmc.hpp"
#include "training/strategies/strategy_factory.hpp"
#include "training_snapshot_test_helpers.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <regex>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

    namespace fs = std::filesystem;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::core::Uuid;
    using Json = lfs::io::JsonChapterDom::Json;
    using namespace lfs::io::project;
    using lfs::test::licht::PENDING_PARAMETER_EXCLUSIONS;
    using lfs::test::licht::require_result;
    using lfs::test::licht::require_result_ptr;
    using lfs::test::licht::TemporaryDirectory;

    Json pending_parameter_field_map(
        const lfs::core::param::OptimizationParameters& parameters,
        const ParameterManagerSnapshot::ReferenceBindings&
            references) {
        Json result =
            Json::parse(parameters.to_json().dump());
        for (const auto& exclusion :
             PENDING_PARAMETER_EXCLUSIONS) {
            result.erase(exclusion.field);
        }
        result.erase("bg_image_path");
        result.erase("ppisp_sidecar_path");
        if (references.background_image_reference) {
            result["background_image_reference_uuid"] =
                references.background_image_reference->to_string();
        }
        if (references.ppisp_reference) {
            result["ppisp_reference_uuid"] =
                references.ppisp_reference->to_string();
        }
        return result;
    }

    Uuid matrix_uuid(const std::uint64_t tag) {
        return lfs::test::licht::fixed_uuid_in_namespace(0x73000000, tag);
    }

    void require_ok(lfs::Result<void> result) {
        lfs::test::licht::require_status(std::move(result));
    }

    template <typename T>
    std::string serialized_bytes(const T& value) {
        std::ostringstream stream;
        value.serialize(stream);
        return stream.str();
    }

    ProjectDocumentSaveOptions matrix_save_options(
        const std::uint64_t tag,
        const std::uint64_t wallclock,
        const std::optional<Uuid>& snapshot_uuid =
            std::nullopt) {
        return ProjectDocumentSaveOptions{
            .commit =
                {
                    .kind = CommitKind::Explicit,
                    .commit_uuid = matrix_uuid(tag),
                    .snapshot_uuid =
                        snapshot_uuid.value_or(
                            matrix_uuid(tag + 1)),
                    .wallclock_unix_ns = wallclock,
                },
            .file_uuid = matrix_uuid(tag + 2),
            .index_compression =
                IndexCompression::StoredForDeterministicTests,
            .disk_reserve_bytes = 0,
        };
    }

    struct MatrixCheckpoint {
        std::vector<std::byte> bytes;
        lfs::core::param::TrainingParameters parameters;
        int iteration = 0;
        std::string bilateral_bytes;
        std::string ppisp_bytes;
        std::string controller_bytes;
        std::string sparsity_bytes;
        lfs::training::test::
            AdamMomentByteSnapshot optimizer_moments;
    };

    MatrixCheckpoint make_matrix_checkpoint() {
        MatrixCheckpoint result;
        result.iteration = 12'345;
        result.parameters.optimization =
            lfs::core::param::OptimizationParameters::
                mcmc_defaults();
        result.parameters.optimization.iterations =
            20'000;
        result.parameters.optimization.max_cap = 8;
        result.parameters.optimization.eval_steps =
            {1'111, 2'222};
        result.parameters.optimization.save_steps =
            {3'333, 4'444};
        result.parameters.optimization.use_depth_loss =
            true;
        result.parameters.optimization.depth_loss_weight =
            0.125f;
        result.parameters.optimization.bg_mode =
            lfs::core::param::BackgroundMode::Random;
        result.parameters.optimization.bg_color =
            {0.125f, 0.25f, 0.5f};
        result.parameters.optimization.use_bilateral_grid =
            true;
        result.parameters.optimization.use_ppisp = true;
        result.parameters.optimization
            .ppisp_use_controller = true;
        result.parameters.optimization.enable_sparsity =
            true;
        result.parameters.optimization.sparsify_steps =
            100;
        result.parameters.optimization.init_rho =
            0.001f;
        result.parameters.optimization.prune_ratio =
            0.25f;
        result.parameters.dataset.images =
            "images_matrix_checkpoint";
        result.parameters.dataset.resize_factor = 4;
        result.parameters.dataset.test_every = 9;
        result.parameters.dataset.timelapse_images =
            {"checkpoint_a.png", "checkpoint_b.png"};
        result.parameters.dataset.timelapse_every = 61;
        result.parameters.dataset.max_width = 1'920;
        result.parameters.dataset.min_track_length = 7;
        result.parameters.dataset.loading_params
            .use_16bit_color = true;
        result.parameters.freeze_lr_scale = 0.25f;
        result.parameters.add_splat_paths = {
            "frozen-source.ply"};
        result.parameters.add_splat_freeze = {true};
        result.parameters
            .exclude_frozen_add_splats_from_export = true;

        auto model =
            lfs::test::licht::make_matrix_splat(true);
        lfs::training::MCMC strategy(*model);
        strategy.initialize(
            result.parameters.optimization);
        {
            const auto parameter_types =
                lfs::training::AdamOptimizer::
                    all_param_types();
            for (std::size_t index = 0;
                 index < parameter_types.size();
                 ++index) {
                auto* state =
                    strategy.get_optimizer()
                        .get_state_mutable(
                            parameter_types[index]);
                EXPECT_NE(state, nullptr);
                if (!state) {
                    continue;
                }
                state->step_count =
                    static_cast<std::int64_t>(
                        100 + index);
                EXPECT_EQ(
                    cudaMemset(
                        state->exp_avg.data_ptr(),
                        static_cast<int>(11 + index),
                        state->exp_avg.bytes()),
                    cudaSuccess);
                EXPECT_EQ(
                    cudaMemset(
                        state->exp_avg_sq.data_ptr(),
                        static_cast<int>(31 + index),
                        state->exp_avg_sq.bytes()),
                    cudaSuccess);
                state->exp_avg_scale.fill_(
                    0.25f +
                    static_cast<float>(index));
                state->exp_avg_sq_scale.fill_(
                    0.5f +
                    static_cast<float>(index));
            }
            EXPECT_EQ(
                cudaDeviceSynchronize(),
                cudaSuccess);
        }
        result.optimizer_moments =
            lfs::training::test::
                capture_optimizer_moment_bytes(
                    strategy.get_optimizer());
        lfs::training::BilateralGrid bilateral(
            1, 2, 3, 4,
            result.parameters.optimization
                .iterations);
        lfs::training::PPISP ppisp(
            result.parameters.optimization
                .iterations);
        ppisp.register_frame(7, 11);
        ppisp.finalize();
        lfs::training::PPISPControllerPool controller(
            1,
            result.parameters.optimization
                .iterations);
        lfs::training::ADMMSparsityOptimizer sparsity({
            .sparsify_steps = 100,
            .init_rho = 0.001f,
            .prune_ratio = 0.25f,
            .update_every = 50,
            .start_iteration = 20,
        });
        const auto initialized_sparsity =
            sparsity.initialize(model->opacity_raw());
        EXPECT_TRUE(initialized_sparsity)
            << (initialized_sparsity
                    ? std::string{}
                    : initialized_sparsity.error());

        result.bilateral_bytes = serialized_bytes(bilateral);
        result.ppisp_bytes = serialized_bytes(ppisp);
        result.controller_bytes = serialized_bytes(controller);
        result.sparsity_bytes = serialized_bytes(sparsity);

        std::ostringstream stream(
            std::ios::binary | std::ios::out);
        const auto serialized =
            lfs::training::serialize_checkpoint(
                stream, result.iteration, strategy,
                result.parameters, &bilateral, &ppisp,
                &controller, &sparsity);
        EXPECT_TRUE(serialized)
            << (serialized ? std::string{}
                           : lfs::format_for_developer(
                                 serialized.error()));
        if (!serialized) {
            return result;
        }
        const std::string bytes = stream.str();
        EXPECT_EQ(bytes.size(), serialized->bytes);
        result.bytes.resize(bytes.size());
        std::memcpy(
            result.bytes.data(), bytes.data(),
            bytes.size());
        return result;
    }

    std::vector<std::byte> copy_lazy_bytes(
        const LazyChunkValue& payload,
        const std::size_t window_bytes = 7) {
        std::ostringstream stream(
            std::ios::binary | std::ios::out);
        const auto copied =
            payload.copy_to(stream, window_bytes);
        EXPECT_TRUE(copied)
            << (copied ? std::string{}
                       : lfs::format_for_developer(
                             copied.error()));
        const std::string bytes = stream.str();
        std::vector<std::byte> result(bytes.size());
        std::memcpy(
            result.data(), bytes.data(),
            bytes.size());
        return result;
    }

    std::vector<std::string> split_markdown_row(
        const std::string_view line) {
        std::vector<std::string> cells;
        std::size_t begin = 0;
        while (begin <= line.size()) {
            const std::size_t end = line.find('|', begin);
            std::string cell(line.substr(
                begin, end == std::string_view::npos
                           ? line.size() - begin
                           : end - begin));
            const auto first = cell.find_first_not_of(" \t");
            const auto last = cell.find_last_not_of(" \t");
            cells.push_back(
                first == std::string::npos
                    ? std::string{}
                    : cell.substr(first, last - first + 1));
            if (end == std::string_view::npos) {
                break;
            }
            begin = end + 1;
        }
        return cells;
    }

    std::string phase_for_authority(const std::string_view authority) {
        if (authority == "PROJ" || authority == "PRMS" ||
            authority == "SCNG" || authority == "SELM" ||
            authority == "REFS" || authority == "SPLT") {
            return "P3";
        }
        if (authority == "CKPT" || authority == "PPIS") {
            return "P4";
        }
        if (authority == "GUIL" || authority == "EDTR" ||
            authority == "VIEW" || authority == "SEQR" ||
            authority == "METR") {
            return "P5";
        }
        return {};
    }

    struct MatrixRows {
        std::set<std::string> p3;
        std::set<std::string> p4;
        std::set<std::string> p5;
        std::set<std::string> exclusions;
    };

    MatrixRows read_matrix_rows() {
        const fs::path path =
            fs::path(PROJECT_ROOT_PATH) / "docs/licht_ownership_matrix.md";
        std::ifstream stream(path);
        EXPECT_TRUE(stream.is_open()) << path;
        MatrixRows result;
        std::string line;
        std::string section;
        std::size_t line_number = 0;
        while (std::getline(stream, line)) {
            ++line_number;
            if (line.starts_with("## `")) {
                const auto close = line.find('`', 4);
                section = close == std::string::npos
                              ? std::string{}
                              : line.substr(4, close - 4);
                continue;
            }
            if (line == "## Exclusions") {
                section = "EXCLUSIONS";
                continue;
            }
            if (line.starts_with("## ")) {
                section.clear();
                continue;
            }
            if (line.empty() || line.front() != '|') {
                continue;
            }
            const auto cells = split_markdown_row(line);
            if (section == "EXCLUSIONS") {
                if (cells.size() < 5 || cells[1] == "Excluded state" ||
                    cells[1].starts_with("---")) {
                    continue;
                }
                EXPECT_FALSE(cells[1].empty()) << "matrix line " << line_number;
                EXPECT_FALSE(cells[2].empty()) << "matrix line " << line_number;
                EXPECT_FALSE(cells[3].empty())
                    << "Every exclusion needs an explicit not-serialized boundary proof at line "
                    << line_number;
                result.exclusions.insert(
                    std::format("EXCLUDED-{}", line_number));
                continue;
            }
            const std::string phase = phase_for_authority(section);
            if (phase.empty() || cells.size() < 7 ||
                cells[1] == "Field" || cells[1].starts_with("---")) {
                continue;
            }
            const std::string expected_authority =
                std::format("`{}`", section);
            EXPECT_EQ(cells[4], expected_authority)
                << "Every persisted-field row has exactly one authority, matching its chapter section, at line "
                << line_number;
            EXPECT_GE(cells[4].size(), 3u);
            if (cells[4].size() < 3u) {
                continue;
            }
            const std::string authority =
                cells[4].substr(1, cells[4].size() - 2);
            EXPECT_EQ(authority, section) << "matrix line " << line_number;
            EXPECT_FALSE(cells[1].empty()) << "matrix line " << line_number;
            EXPECT_FALSE(cells[2].empty()) << "matrix line " << line_number;
            EXPECT_FALSE(cells[5].empty()) << "matrix line " << line_number;
            EXPECT_FALSE(cells[6].empty()) << "matrix line " << line_number;
            const std::string row_id =
                std::format("{}-{}", authority, line_number);
            if (phase == "P3") {
                result.p3.insert(row_id);
            } else if (phase == "P4") {
                result.p4.insert(row_id);
            } else {
                result.p5.insert(row_id);
            }
        }
        return result;
    }

    std::string ownership_item(
        const std::string_view chapter,
        const std::string_view kind,
        const std::string_view path) {
        return std::format("{}|{}|{}", chapter, kind, path);
    }

    void collect_json_ownership_paths(
        const Json& value,
        const std::string& prefix,
        std::set<std::string>& paths) {
        if (value.is_object()) {
            for (const auto& [key, child] : value.items()) {
                const std::string path =
                    prefix.empty() ? key : std::format("{}.{}", prefix, key);
                paths.insert(path);
                collect_json_ownership_paths(child, path, paths);
            }
            return;
        }
        if (!value.is_array()) {
            return;
        }
        const std::string elements = std::format("{}[]", prefix);
        paths.insert(elements);
        for (const auto& child : value) {
            collect_json_ownership_paths(child, elements, paths);
        }
    }

    std::set<std::string> runtime_ownership_inventory(
        const fs::path& project_path) {
        auto reader = ProjectReader::open(project_path);
        if (!reader) {
            throw std::runtime_error(
                lfs::format_for_developer(reader.error()));
        }
        const std::set<std::string> json_chapters{
            "PROJ",
            "PRMS",
            "SCNG",
            "REFS",
            "GUIL",
            "EDTR",
            "VIEW",
            "SEQR",
        };
        std::set<std::string> inventory;
        for (const auto& row : reader->chunks()) {
            if (row.row_kind != RowKind::Live) {
                continue;
            }
            const std::string chapter = row.key.fourcc.to_string();
            inventory.insert(ownership_item(chapter, "chunk", chapter));
            if (!json_chapters.contains(chapter)) {
                continue;
            }
            auto payload = reader->read_chunk(row);
            if (!payload) {
                throw std::runtime_error(
                    lfs::format_for_developer(payload.error()));
            }
            const auto text = std::string(
                reinterpret_cast<const char*>(payload->data()),
                payload->size());
            const Json root = Json::parse(text);
            std::set<std::string> paths;
            collect_json_ownership_paths(root, {}, paths);
            for (auto path : paths) {
                if (chapter == "PRMS" && path.starts_with("presets.")) {
                    const auto strategy_end = path.find('.', 8);
                    if (strategy_end != std::string::npos) {
                        path.replace(8, strategy_end - 8, "*");
                    }
                }
                inventory.insert(ownership_item(chapter, "json", path));
            }
        }
        return inventory;
    }

    void assert_runtime_ownership_inventory(
        const fs::path& project_path) {
        const auto observed = runtime_ownership_inventory(project_path);
        const fs::path matrix_path =
            fs::path(PROJECT_ROOT_PATH) / "docs/licht_ownership_matrix.md";
        std::ifstream matrix(matrix_path);
        ASSERT_TRUE(matrix.is_open()) << matrix_path;
        const std::regex marker(
            R"(<!-- P8-RUNTIME chapter=([A-Z0-9]{4}) kind=(chunk|json) path=([^ ]+) authority=([A-Z0-9]{4}) -->)");
        std::map<std::string, std::set<std::string>> declarations;
        std::string line;
        while (std::getline(matrix, line)) {
            std::smatch match;
            if (!std::regex_match(line, match, marker)) {
                continue;
            }
            declarations[ownership_item(match[1].str(), match[2].str(),
                                        match[3].str())]
                .insert(match[4].str());
        }
        std::set<std::string> documented;
        for (const auto& [item, authorities] : declarations) {
            EXPECT_EQ(authorities.size(), 1u)
                << "A serialized item must have exactly one authority: " << item;
            const auto chapter = item.substr(0, item.find('|'));
            EXPECT_TRUE(authorities.contains(chapter))
                << "Serialized item authority must match its chapter: " << item;
            documented.insert(item);
        }
        EXPECT_EQ(documented, observed)
            << "Runtime serialization added, removed, or renamed a field without updating the ownership inventory";
    }

    TEST(P8OwnershipMatrixRatchet,
         MaximallyPopulatedSerializedOutputMatchesOwnershipMatrix) {
        TemporaryDirectory temporary{"lfs-p3-matrix"};
        const fs::path path = temporary.path / "matrix-proof.licht";

        auto fixture = lfs::test::licht::make_populated_project_fixture();
        auto& document = fixture.document;
        const auto& project_uuid = fixture.project_uuid;
        const auto& dataset_ref = fixture.dataset_reference;
        const auto& background_ref = fixture.background_reference;
        const auto& ppisp_ref = fixture.ppisp_reference;
        const auto& root_node = fixture.root_node;
        const auto& training_node = fixture.training_node;
        const auto& imported_node = fixture.imported_node;
        const auto& point_node = fixture.point_node;
        const auto& mesh_node = fixture.mesh_node;
        const auto& crop_node = fixture.crop_node;
        const auto& ellipsoid_node = fixture.ellipsoid_node;
        const auto& camera_node = fixture.camera_node;
        const auto& checkpoint_uuid = fixture.checkpoint_uuid;
        const auto& manifest = fixture.manifest;
        const auto& georeference = fixture.georeference;
        const auto& edited_transform = fixture.edited_transform;
        const auto& camera = fixture.camera;
        const auto& expected_references = fixture.references;
        const auto& expected_nodes = fixture.nodes;
        const auto& parameters = fixture.parameters;

        const auto expected_checkpoint =
            make_matrix_checkpoint();
        ASSERT_FALSE(expected_checkpoint.bytes.empty());
        auto checkpoint_payload = require_result(LazyChunkValue::from_owned(
            std::make_shared<const std::vector<std::byte>>(expected_checkpoint.bytes),
            checkpoint_uuid));
        require_ok(document->set_checkpoint(checkpoint_uuid, std::move(checkpoint_payload)));

        auto first_save = require_result_ptr(
            document->save(path, matrix_save_options(100, 200, checkpoint_uuid)));
        assert_runtime_ownership_inventory(path);

        auto first_open = require_result_ptr(ProjectDocument::open(path));
        const auto first_refs_bytes = first_open->references().to_bytes();
        const auto first_scng_bytes = first_open->scene_graph().to_bytes();
        const auto first_selm_bytes =
            encode_selection_chapter(first_open->selection());
        ASSERT_TRUE(first_selm_bytes);
        const auto first_prms_bytes = first_open->parameters().to_bytes();
        const auto* first_splat = first_open->find_splat(imported_node);
        const auto* first_point =
            first_open->find_point_cloud(point_node);
        const auto* first_mesh = first_open->find_mesh(mesh_node);
        ASSERT_NE(first_splat, nullptr);
        ASSERT_NE(first_point, nullptr);
        ASSERT_NE(first_mesh, nullptr);
        const auto* first_checkpoint =
            first_open->find_checkpoint(checkpoint_uuid);
        ASSERT_NE(first_checkpoint, nullptr);
        EXPECT_TRUE(first_checkpoint->is_clean_reference());
        const auto first_checkpoint_bytes =
            copy_lazy_bytes(*first_checkpoint);
        EXPECT_EQ(first_checkpoint_bytes, expected_checkpoint.bytes);
        const std::vector<std::byte> first_splat_bytes(
            first_splat->bytes().begin(), first_splat->bytes().end());
        const auto first_point_bytes =
            encode_point_cloud_payload(*first_point);
        const auto first_mesh_bytes = encode_mesh_payload(*first_mesh);
        ASSERT_TRUE(first_point_bytes);
        ASSERT_TRUE(first_mesh_bytes);

        auto second_save = require_result_ptr(
            first_open->save(path, matrix_save_options(110, 300, checkpoint_uuid)));
        EXPECT_EQ(second_save->rewritten_chunks, 1u);
        EXPECT_EQ(second_save->reused_chunks, 13u);

        auto second_open = require_result_ptr(ProjectDocument::open(path));
        std::set<std::string> proven;
        const auto prove = [&](const std::string_view row) {
            EXPECT_TRUE(proven.emplace(row).second) << row;
        };

        auto second_manifest = second_open->project().manifest();
        ASSERT_TRUE(second_manifest);
        EXPECT_EQ(*second_manifest, manifest);
        prove("PROJ-39");
        auto second_project_uuid = second_open->project().project_uuid();
        ASSERT_TRUE(second_project_uuid);
        EXPECT_EQ(*second_project_uuid, project_uuid);
        prove("PROJ-40");
        auto created = second_open->project().created_at_unix_ns();
        auto modified = second_open->project().modified_at_unix_ns();
        ASSERT_TRUE(created);
        ASSERT_TRUE(modified);
        EXPECT_EQ(*created, 100u);
        EXPECT_EQ(*modified, 300u);
        prove("PROJ-41");
        auto reopened_dataset =
            second_open->project().dataset_reference();
        ASSERT_TRUE(reopened_dataset);
        EXPECT_EQ(*reopened_dataset, dataset_ref);
        prove("PROJ-42");
        auto decisions = second_open->project().embed_decisions();
        ASSERT_TRUE(decisions);
        EXPECT_EQ(decisions->size(), 4u);
        EXPECT_EQ(std::ranges::count_if(
                      *decisions,
                      [](const EmbedDecision& item) {
                          return item.decision == "embedded";
                      }),
                  3);
        EXPECT_EQ(std::ranges::count_if(
                      *decisions,
                      [](const EmbedDecision& item) {
                          return item.decision == "external";
                      }),
                  1);
        prove("PROJ-43");
        auto source_provenance = second_open->project().provenance();
        auto embedded_provenance =
            second_open->project().embedded_payload_provenance();
        ASSERT_TRUE(source_provenance);
        ASSERT_TRUE(embedded_provenance);
        EXPECT_EQ(source_provenance->size(), 2u);
        ASSERT_EQ(embedded_provenance->size(), 3u);
        EXPECT_TRUE(std::ranges::all_of(
            *embedded_provenance,
            [](const EmbeddedPayloadProvenance& item) {
                return !item.import_locator.preferred.empty() &&
                       item.import_fingerprint.full_xxh3.has_value() &&
                       !item.content_xxh3_128.is_zero();
            }));
        prove("PROJ-44");
        auto reopened_georeference =
            second_open->project().georeference();
        ASSERT_TRUE(reopened_georeference);
        EXPECT_EQ(*reopened_georeference, georeference);
        prove("PROJ-45");

        EXPECT_EQ(second_open->references().to_bytes(),
                  first_refs_bytes);
        auto reopened_references =
            second_open->references().records();
        ASSERT_TRUE(reopened_references);
        EXPECT_EQ(*reopened_references, expected_references);
        prove("REFS-96");
        const auto prove_reference_records =
            [&](const std::string_view row,
                const std::initializer_list<std::string_view>
                    kinds) {
                const auto selected =
                    [&](const ReferenceRecord& record) {
                        return std::ranges::find(
                                   kinds, record.kind) !=
                               kinds.end();
                    };
                std::vector<ReferenceRecord> expected;
                std::vector<ReferenceRecord> actual;
                std::ranges::copy_if(
                    expected_references,
                    std::back_inserter(expected), selected);
                std::ranges::copy_if(
                    *reopened_references,
                    std::back_inserter(actual), selected);
                EXPECT_FALSE(expected.empty()) << row;
                EXPECT_EQ(actual, expected) << row;
                prove(row);
            };
        prove_reference_records("REFS-97", {"dataset"});
        prove_reference_records("REFS-98", {"colmap"});
        prove_reference_records(
            "REFS-99", {"rad", "rad_meta_cache"});
        prove_reference_records(
            "REFS-100", {"background_image"});
        prove_reference_records(
            "REFS-101", {"environment_map"});
        prove_reference_records(
            "REFS-102", {"ply_sequence"});

        EXPECT_EQ(second_open->scene_graph().to_bytes(),
                  first_scng_bytes);
        auto reopened_nodes = second_open->scene_graph().nodes();
        ASSERT_TRUE(reopened_nodes);
        EXPECT_EQ(*reopened_nodes, expected_nodes);
        EXPECT_TRUE(std::ranges::all_of(
            *reopened_nodes,
            [](const SceneNodeRecord& node) {
                return !node.uuid.is_nil();
            }));
        prove("SCNG-64");
        EXPECT_TRUE(std::ranges::all_of(
            *reopened_nodes,
            [](const SceneNodeRecord& node) {
                return !node.type.empty() && !node.name.empty();
            }));
        prove("SCNG-65");
        const auto reopened_imported =
            second_open->scene_graph().find(imported_node);
        const auto reopened_point =
            second_open->scene_graph().find(point_node);
        const auto reopened_crop =
            second_open->scene_graph().find(crop_node);
        const auto reopened_ellipsoid =
            second_open->scene_graph().find(ellipsoid_node);
        const auto reopened_camera =
            second_open->scene_graph().find(camera_node);
        ASSERT_TRUE(reopened_imported && *reopened_imported);
        ASSERT_TRUE(reopened_point && *reopened_point);
        ASSERT_TRUE(reopened_crop && *reopened_crop);
        ASSERT_TRUE(reopened_ellipsoid && *reopened_ellipsoid);
        ASSERT_TRUE(reopened_camera && *reopened_camera);
        EXPECT_EQ((*reopened_imported)->parent_uuid, root_node);
        EXPECT_EQ((*reopened_imported)->child_order, 3u);
        prove("SCNG-66");
        EXPECT_EQ((*reopened_imported)->local_transform,
                  edited_transform);
        prove("SCNG-67");
        ASSERT_TRUE((*reopened_point)->georef_pose);
        EXPECT_EQ((*reopened_point)->georef_pose->translation,
                  (std::array<double, 3>{1000.25, 2000.5, -10.75}));
        prove("SCNG-68");
        EXPECT_FALSE((*reopened_imported)->visible);
        EXPECT_TRUE((*reopened_imported)->locked);
        prove("SCNG-69");
        EXPECT_TRUE((*reopened_imported)->payload_diverged);
        prove("SCNG-70");
        auto reopened_training =
            second_open->scene_graph().training_model_uuid();
        ASSERT_TRUE(reopened_training);
        EXPECT_EQ(*reopened_training, training_node);
        prove("SCNG-71");
        EXPECT_FALSE((*reopened_camera)->training_enabled);
        prove("SCNG-72");
        ASSERT_TRUE((*reopened_crop)->cropbox);
        EXPECT_EQ((*reopened_crop)->cropbox,
                  expected_nodes[11].cropbox);
        prove("SCNG-73");
        ASSERT_TRUE((*reopened_ellipsoid)->ellipsoid);
        EXPECT_EQ((*reopened_ellipsoid)->ellipsoid,
                  expected_nodes[12].ellipsoid);
        prove("SCNG-74");
        ASSERT_TRUE((*reopened_camera)->camera);
        EXPECT_EQ((*reopened_camera)->camera->uid, 71);
        EXPECT_EQ((*reopened_camera)->camera->camera_id, 72);
        EXPECT_EQ((*reopened_camera)->camera->rotation,
                  camera.rotation);
        EXPECT_FLOAT_EQ((*reopened_camera)->camera->focal_x,
                        1200.5f);
        EXPECT_EQ((*reopened_camera)->camera->radial_distortion,
                  camera.radial_distortion);
        prove("SCNG-75");
        EXPECT_EQ((*reopened_camera)->camera->image_name,
                  "frame_0071.png");
        EXPECT_EQ((*reopened_camera)->camera->image_path,
                  "images/frame_0071.png");
        EXPECT_EQ((*reopened_camera)->camera->normal_path,
                  "normals/frame_0071.png");
        EXPECT_TRUE((*reopened_camera)->camera->has_alpha);
        EXPECT_EQ((*reopened_camera)->camera->split, "eval");
        prove("SCNG-76");
        const std::set<std::string> structural_types{
            "group", "dataset", "camera_group", "image_group",
            "image", "ply_sequence"};
        EXPECT_TRUE(std::ranges::all_of(
            structural_types,
            [&](const std::string& type) {
                return std::ranges::any_of(
                    *reopened_nodes,
                    [&](const SceneNodeRecord& node) {
                        return node.type == type;
                    });
            }));
        EXPECT_EQ((*reopened_imported)->payload->fourcc, "SPLT");
        EXPECT_EQ((*reopened_point)->payload->fourcc, "PCLD");
        prove("SCNG-77");

        auto second_selm_bytes =
            encode_selection_chapter(second_open->selection());
        ASSERT_TRUE(second_selm_bytes);
        EXPECT_EQ(*second_selm_bytes, *first_selm_bytes);
        ASSERT_EQ(second_open->selection().groups().size(), 2u);
        EXPECT_EQ(second_open->selection().groups()[1].name,
                  "Protected");
        EXPECT_TRUE(second_open->selection().groups()[1].locked);
        prove("SELM-83");
        EXPECT_EQ(second_open->selection().active_group_id(), 7u);
        EXPECT_EQ(second_open->selection().next_group_id(), 8u);
        prove("SELM-84");
        ASSERT_EQ(second_open->selection().slices().size(), 2u);
        EXPECT_TRUE(std::ranges::any_of(
            second_open->selection().slices(),
            [](const SelectionMaskSlice& slice) {
                return slice.domain ==
                           lfs::core::SelectionDomain::Splat &&
                       slice.mask == std::vector<std::uint8_t>{3, 0, 7, 3};
            }));
        EXPECT_TRUE(std::ranges::any_of(
            second_open->selection().slices(),
            [](const SelectionMaskSlice& slice) {
                return slice.domain ==
                           lfs::core::SelectionDomain::PointCloud &&
                       slice.mask == std::vector<std::uint8_t>{0, 7};
            }));
        prove("SELM-85");
        EXPECT_EQ(second_open->selection().selected_node_uuids(),
                  (std::vector<Uuid>{
                      point_node, imported_node, camera_node}));
        prove("SELM-86");

        EXPECT_EQ(second_open->parameters().to_bytes(),
                  first_prms_bytes);
        auto reopened_parameters =
            second_open->parameters().snapshot();
        ASSERT_TRUE(reopened_parameters);
        EXPECT_EQ(reopened_parameters->active_strategy,
                  lfs::core::param::kStrategyIGSPlus);
        prove("PRMS-53");
        struct PendingRoleProof {
            std::string_view path;
            const lfs::core::param::OptimizationParameters* expected_parameters;
            const lfs::core::param::OptimizationParameters* actual_parameters;
            const ParameterManagerSnapshot::ReferenceBindings* expected_references;
            const ParameterManagerSnapshot::ReferenceBindings* actual_references;
        };
        const std::array pending_roles{
            PendingRoleProof{"presets.mcmc.session", &parameters.mcmc_session,
                             &reopened_parameters->mcmc_session,
                             &parameters.mcmc_session_references,
                             &reopened_parameters->mcmc_session_references},
            PendingRoleProof{"presets.mrnf.session", &parameters.mrnf_session,
                             &reopened_parameters->mrnf_session,
                             &parameters.mrnf_session_references,
                             &reopened_parameters->mrnf_session_references},
            PendingRoleProof{"presets.igs+.session", &parameters.igs_session,
                             &reopened_parameters->igs_session,
                             &parameters.igs_session_references,
                             &reopened_parameters->igs_session_references},
            PendingRoleProof{"presets.mcmc.current", &parameters.mcmc_current,
                             &reopened_parameters->mcmc_current,
                             &parameters.mcmc_current_references,
                             &reopened_parameters->mcmc_current_references},
            PendingRoleProof{"presets.mrnf.current", &parameters.mrnf_current,
                             &reopened_parameters->mrnf_current,
                             &parameters.mrnf_current_references,
                             &reopened_parameters->mrnf_current_references},
            PendingRoleProof{"presets.igs+.current", &parameters.igs_current,
                             &reopened_parameters->igs_current,
                             &parameters.igs_current_references,
                             &reopened_parameters->igs_current_references},
        };
        for (const auto& role : pending_roles) {
            SCOPED_TRACE(role.path);
            const auto serialized =
                second_open->parameters().dom().get_json(
                    role.path);
            ASSERT_TRUE(serialized);
            const auto expected_fields =
                pending_parameter_field_map(
                    *role.expected_parameters,
                    *role.expected_references);
            EXPECT_EQ(*serialized, expected_fields);
            std::set<std::string> serialized_fields;
            for (const auto& [field, ignored] :
                 serialized->items()) {
                (void)ignored;
                serialized_fields.insert(field);
            }
            const auto expected_field_inventory = lfs::test::licht::word_set(
                lfs::test::licht::OPTIMIZATION_PARAMETER_FIELD_DATA);
            EXPECT_EQ(serialized_fields,
                      expected_field_inventory);
            EXPECT_EQ(
                pending_parameter_field_map(
                    *role.actual_parameters,
                    *role.actual_references),
                expected_fields);
            EXPECT_EQ(*role.actual_references,
                      *role.expected_references);
            for (const auto& exclusion :
                 PENDING_PARAMETER_EXCLUSIONS) {
                SCOPED_TRACE(exclusion.phase);
                EXPECT_FALSE(
                    serialized->contains(exclusion.field))
                    << exclusion.field;
            }
        }
        auto reverse_index =
            second_open->reverse_reference_index();
        ASSERT_TRUE(reverse_index);
        EXPECT_TRUE(std::ranges::any_of(
            reverse_index->at(background_ref),
            [](const ReferenceOwnerBinding& binding) {
                return binding.chapter == "PRMS" &&
                       binding.field ==
                           "presets.mrnf.current."
                           "background_image_reference_uuid";
            }));
        EXPECT_TRUE(std::ranges::any_of(
            reverse_index->at(ppisp_ref),
            [](const ReferenceOwnerBinding& binding) {
                return binding.chapter == "PRMS" &&
                       binding.field ==
                           "presets.mrnf.session."
                           "ppisp_reference_uuid";
            }));
        prove("PRMS-54");
        EXPECT_EQ(reopened_parameters->dataset.images,
                  "images_matrix");
        EXPECT_EQ(reopened_parameters->dataset.timelapse_images,
                  (std::vector<std::string>{
                      "frame_a.png", "frame_b.png"}));
        EXPECT_EQ(reopened_parameters->dataset.centralize_dataset,
                  "cameras");
        EXPECT_TRUE(reopened_parameters->dataset.invert_masks);
        prove("PRMS-55");
        EXPECT_FALSE(
            reopened_parameters->dataset.loading_params.use_cpu_memory);
        EXPECT_FLOAT_EQ(
            reopened_parameters->dataset.loading_params.min_cpu_free_GB,
            3.5f);
        EXPECT_FALSE(
            reopened_parameters->dataset.loading_params.use_fs_cache);
        EXPECT_EQ(
            reopened_parameters->dataset.loading_params
                .print_status_freq_num,
            123);
        EXPECT_TRUE(
            reopened_parameters->dataset.loading_params.use_16bit_color);
        prove("PRMS-56");

        const auto* second_splat =
            second_open->find_splat(imported_node);
        ASSERT_NE(second_splat, nullptr);
        EXPECT_TRUE(std::ranges::equal(
            second_splat->bytes(), first_splat_bytes));
        auto hydrated_splat = second_splat->hydrate();
        ASSERT_TRUE(hydrated_splat)
            << lfs::format_for_developer(hydrated_splat.error());
        EXPECT_EQ(second_splat->lfsp_version(), 4u);
        EXPECT_EQ((*hydrated_splat)->get_active_sh_degree(), 0);
        EXPECT_EQ((*hydrated_splat)->get_max_sh_degree(), 1);
        EXPECT_FLOAT_EQ((*hydrated_splat)->get_scene_scale(), 2.5f);
        prove("SPLT-110");
        EXPECT_EQ((*hydrated_splat)->means().shape().str(), "[4, 3]");
        EXPECT_FLOAT_EQ(
            (*hydrated_splat)->means().cpu().ptr<float>()[0], 1.0f);
        prove("SPLT-111");
        EXPECT_EQ((*hydrated_splat)->sh0().shape().str(),
                  "[4, 1, 3]");
        EXPECT_TRUE((*hydrated_splat)->shN().is_valid());
        prove("SPLT-112");
        EXPECT_EQ((*hydrated_splat)->scaling_raw().shape().str(),
                  "[4, 3]");
        EXPECT_EQ((*hydrated_splat)->rotation_raw().shape().str(),
                  "[4, 4]");
        EXPECT_EQ((*hydrated_splat)->opacity_raw().shape().str(),
                  "[4, 1]");
        prove("SPLT-113");
        EXPECT_TRUE((*hydrated_splat)->deleted().is_valid());
        EXPECT_EQ(
            (*hydrated_splat)->deleted().cpu().to_vector_uint8(),
            (std::vector<std::uint8_t>{0, 1, 0, 1}));
        prove("SPLT-114");
        EXPECT_EQ(
            (*hydrated_splat)->_densification_info.shape().str(),
            "[2, 4]");
        prove("SPLT-115");
        ASSERT_EQ((*hydrated_splat)->frozen_ranges().size(), 1u);
        EXPECT_EQ((*hydrated_splat)->frozen_ranges()[0].start, 1u);
        EXPECT_EQ((*hydrated_splat)->frozen_ranges()[0].count, 2u);
        prove("SPLT-116");

        const auto* second_point =
            second_open->find_point_cloud(point_node);
        const auto* second_mesh =
            second_open->find_mesh(mesh_node);
        ASSERT_NE(second_point, nullptr);
        ASSERT_NE(second_mesh, nullptr);
        auto second_point_bytes =
            encode_point_cloud_payload(*second_point);
        auto second_mesh_bytes = encode_mesh_payload(*second_mesh);
        ASSERT_TRUE(second_point_bytes);
        ASSERT_TRUE(second_mesh_bytes);
        EXPECT_EQ(*second_point_bytes, *first_point_bytes);
        EXPECT_EQ(second_point->point_cloud()->means.shape().str(),
                  "[2, 3]");
        EXPECT_TRUE(second_point->point_cloud()->normals.is_valid());
        EXPECT_TRUE(second_point->point_cloud()->sh0.is_valid());
        EXPECT_TRUE(second_point->point_cloud()->shN.is_valid());
        EXPECT_TRUE(second_point->point_cloud()->opacity.is_valid());
        EXPECT_TRUE(second_point->point_cloud()->scaling.is_valid());
        EXPECT_TRUE(second_point->point_cloud()->rotation.is_valid());
        EXPECT_EQ(second_point->point_cloud()->attribute_names.size(),
                  7u);
        const auto point_vendor = std::ranges::find(
            second_point->retained_properties(),
            std::string_view{"vendor_score"},
            &GeometryPropertyPlane::name);
        ASSERT_NE(point_vendor,
                  second_point->retained_properties().end());
        EXPECT_EQ(point_vendor->bytes,
                  lfs::test::licht::byte_values({0x10, 0x20, 0x30, 0x40}));

        EXPECT_EQ(*second_mesh_bytes, *first_mesh_bytes);
        EXPECT_TRUE(second_mesh->mesh()->normals.is_valid());
        EXPECT_TRUE(second_mesh->mesh()->tangents.is_valid());
        EXPECT_TRUE(second_mesh->mesh()->texcoords.is_valid());
        EXPECT_TRUE(second_mesh->mesh()->colors.is_valid());
        ASSERT_EQ(second_mesh->mesh()->submeshes.size(), 1u);
        EXPECT_EQ(second_mesh->mesh()->submeshes[0].index_count, 6u);
        ASSERT_EQ(second_mesh->mesh()->materials.size(), 1u);
        EXPECT_EQ(second_mesh->mesh()->materials[0].name,
                  "matrix-plane");
        EXPECT_TRUE(second_mesh->mesh()->materials[0].double_sided);
        ASSERT_EQ(second_mesh->mesh()->texture_images.size(), 1u);
        EXPECT_EQ(second_mesh->mesh()->texture_images[0].pixels,
                  (std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6}));
        const auto mesh_vendor = std::ranges::find(
            second_mesh->retained_properties(),
            std::string_view{"vendor_ids"},
            &GeometryPropertyPlane::name);
        ASSERT_NE(mesh_vendor,
                  second_mesh->retained_properties().end());

        std::set<std::string> proven_ckpt;
        const auto prove_ckpt = [&](const std::string_view row) {
            EXPECT_TRUE(proven_ckpt.emplace(row).second) << row;
        };
        const auto* second_checkpoint = second_open->find_checkpoint(checkpoint_uuid);
        ASSERT_NE(second_checkpoint, nullptr);
        EXPECT_TRUE(second_checkpoint->is_clean_reference());
        EXPECT_FALSE(second_checkpoint->owns_staged_bytes());
        EXPECT_EQ(copy_lazy_bytes(*second_checkpoint), expected_checkpoint.bytes);

        std::optional<std::expected<lfs::core::CheckpointHeader, std::string>> reopened_header;
        std::optional<std::expected<lfs::core::SplatData, std::string>> reopened_display_model;
        std::optional<std::expected<lfs::core::param::TrainingParameters, std::string>>
            reopened_checkpoint_parameters;
        std::string reopened_strategy_type;
        require_ok(second_checkpoint->visit_stream(
            [&](std::istream& stream, const std::uint64_t bytes) -> lfs::Result<void> {
                reopened_header = lfs::core::load_checkpoint_header(stream, bytes);
                stream.clear();
                stream.seekg(sizeof(lfs::core::CheckpointHeader), std::ios::beg);
                std::uint32_t strategy_size = 0;
                stream.read(reinterpret_cast<char*>(&strategy_size), sizeof(strategy_size));
                if (stream && strategy_size <= lfs::core::MAX_CHECKPOINT_STRATEGY_NAME_BYTES) {
                    reopened_strategy_type.resize(strategy_size);
                    stream.read(reopened_strategy_type.data(), strategy_size);
                }
                stream.clear();
                stream.seekg(0);
                reopened_display_model = lfs::core::load_checkpoint_splat_data(stream, bytes);
                stream.clear();
                stream.seekg(0);
                reopened_checkpoint_parameters = lfs::core::load_checkpoint_params(stream, bytes);
                return {};
            }));
        ASSERT_TRUE(reopened_header);
        ASSERT_TRUE(*reopened_header) << reopened_header->error();
        const auto& header = **reopened_header;
        EXPECT_EQ(header.iteration, expected_checkpoint.iteration);
        EXPECT_EQ(header.num_gaussians, 4u);
        EXPECT_EQ(header.version, lfs::core::CHECKPOINT_VERSION);
        prove_ckpt("CKPT-126");

        ASSERT_TRUE(reopened_display_model);
        ASSERT_TRUE(*reopened_display_model) << reopened_display_model->error();
        EXPECT_EQ((**reopened_display_model).size(), 4u);
        ASSERT_EQ((**reopened_display_model).frozen_ranges().size(), 1u);
        prove_ckpt("CKPT-127");

        EXPECT_EQ(reopened_strategy_type, lfs::core::param::kStrategyMCMC);
        prove_ckpt("CKPT-128");

        auto target_model =
            lfs::test::licht::make_matrix_splat(true);
        lfs::training::MCMC target_strategy(*target_model);
        auto restored_parameters = expected_checkpoint.parameters;
        target_strategy.initialize(restored_parameters.optimization);
        lfs::training::BilateralGrid restored_bilateral(1, 1, 1, 1, 1);
        lfs::training::PPISP restored_ppisp(1);
        lfs::training::PPISPControllerPool restored_controller(1, 1);
        lfs::training::ADMMSparsityOptimizer restored_sparsity({
            .sparsify_steps = 1,
            .init_rho = 0.5f,
            .prune_ratio = 0.9f,
            .update_every = 1,
            .start_iteration = 0,
        });
        std::optional<std::expected<int, std::string>> fully_restored;
        require_ok(second_checkpoint->visit_stream(
            [&](std::istream& stream, const std::uint64_t bytes) -> lfs::Result<void> {
                fully_restored = lfs::training::load_checkpoint(
                    stream, bytes, target_strategy, restored_parameters, &restored_bilateral,
                    &restored_ppisp, &restored_controller, &restored_sparsity, {}, "matrix CKPT");
                return {};
            }));
        ASSERT_TRUE(fully_restored);
        ASSERT_TRUE(*fully_restored) << fully_restored->error();
        EXPECT_EQ(**fully_restored, expected_checkpoint.iteration);
        lfs::training::test::expect_optimizer_moment_bytes_equal(
            expected_checkpoint.optimizer_moments, target_strategy.get_optimizer());
        prove_ckpt("CKPT-129");
        EXPECT_NE(target_strategy.get_scheduler(), nullptr);
        prove_ckpt("CKPT-130");
        EXPECT_STREQ(target_strategy.strategy_type(), lfs::core::param::kStrategyMCMC.data());
        prove_ckpt("CKPT-131");

        auto& strategy_factory = lfs::training::StrategyFactory::instance();
        const auto registered_strategies = strategy_factory.list();
        auto sorted_strategies = registered_strategies;
        std::ranges::sort(sorted_strategies);
        EXPECT_EQ(sorted_strategies,
                  (std::vector<std::string>{
                      std::string(lfs::core::param::kStrategyIGSPlus),
                      std::string(lfs::core::param::kStrategyMCMC),
                      std::string(lfs::core::param::kStrategyMRNF),
                  }));
        const auto prove_registered_strategy = [&](const std::string_view strategy_name,
                                                   lfs::core::param::OptimizationParameters optimization,
                                                   const std::string_view row) {
            auto source_model = lfs::test::licht::make_matrix_splat(true);
            auto source_result = strategy_factory.create(
                std::string(strategy_name), *source_model);
            ASSERT_TRUE(source_result) << source_result.error();
            auto source = std::move(*source_result);
            optimization.strategy = std::string(strategy_name);
            optimization.max_cap = 8;
            optimization.sh_degree = 1;
            source->initialize(optimization);

            lfs::core::param::TrainingParameters strategy_parameters;
            strategy_parameters.optimization = optimization;
            std::ostringstream strategy_stream(
                std::ios::binary | std::ios::out);
            const auto serialized_strategy = lfs::training::serialize_checkpoint(
                strategy_stream, expected_checkpoint.iteration, *source,
                strategy_parameters, nullptr, nullptr, nullptr, nullptr);
            ASSERT_TRUE(serialized_strategy)
                << lfs::format_for_developer(serialized_strategy.error());
            const auto strategy_bytes = strategy_stream.str();
            ASSERT_EQ(strategy_bytes.size(), serialized_strategy->bytes);

            auto target_model = lfs::test::licht::make_matrix_splat(true);
            auto target_result = strategy_factory.create(
                std::string(strategy_name), *target_model);
            ASSERT_TRUE(target_result) << target_result.error();
            auto target = std::move(*target_result);
            target->initialize(optimization);
            auto loaded_parameters = strategy_parameters;
            std::istringstream input(strategy_bytes, std::ios::binary | std::ios::in);
            const auto loaded_strategy = lfs::training::load_checkpoint(
                input, strategy_bytes.size(), *target, loaded_parameters,
                nullptr, nullptr, nullptr, nullptr, {}, "matrix strategy CKPT");
            ASSERT_TRUE(loaded_strategy) << loaded_strategy.error();
            EXPECT_EQ(*loaded_strategy, expected_checkpoint.iteration);
            EXPECT_EQ(std::string_view(target->strategy_type()), strategy_name);
            EXPECT_NE(target->get_optimizer().get_state(lfs::training::ParamType::Means),
                      nullptr);
            prove_ckpt(row);
        };
        prove_registered_strategy(
            lfs::core::param::kStrategyMRNF,
            lfs::core::param::OptimizationParameters::mrnf_defaults(),
            "CKPT-132");
        prove_registered_strategy(
            lfs::core::param::kStrategyIGSPlus,
            lfs::core::param::OptimizationParameters::igs_plus_defaults(),
            "CKPT-133");

        EXPECT_TRUE(lfs::core::has_flag(
            header.flags, lfs::core::CheckpointFlags::HAS_BILATERAL_GRID));
        EXPECT_EQ(serialized_bytes(restored_bilateral), expected_checkpoint.bilateral_bytes);
        prove_ckpt("CKPT-134");
        EXPECT_TRUE(lfs::core::has_flag(
            header.flags, lfs::core::CheckpointFlags::HAS_PPISP));
        EXPECT_EQ(serialized_bytes(restored_ppisp), expected_checkpoint.ppisp_bytes);
        prove_ckpt("CKPT-135");
        EXPECT_TRUE(lfs::core::has_flag(
            header.flags, lfs::core::CheckpointFlags::HAS_PPISP_CONTROLLER));
        EXPECT_EQ(serialized_bytes(restored_controller), expected_checkpoint.controller_bytes);
        prove_ckpt("CKPT-136");
        EXPECT_TRUE(lfs::core::has_flag(
            header.flags, lfs::core::CheckpointFlags::HAS_SPARSITY));
        EXPECT_EQ(serialized_bytes(restored_sparsity), expected_checkpoint.sparsity_bytes);
        prove_ckpt("CKPT-137");

        ASSERT_TRUE(reopened_checkpoint_parameters);
        ASSERT_TRUE(*reopened_checkpoint_parameters)
            << reopened_checkpoint_parameters->error();
        const auto& reopened_active = **reopened_checkpoint_parameters;
        const auto& expected_active = expected_checkpoint.parameters;
        EXPECT_EQ(reopened_active.optimization.iterations,
                  expected_active.optimization.iterations);
        EXPECT_FLOAT_EQ(reopened_active.optimization.means_lr,
                        expected_active.optimization.means_lr);
        prove_ckpt("CKPT-138");
        EXPECT_EQ(reopened_active.optimization.eval_steps,
                  expected_active.optimization.eval_steps);
        EXPECT_EQ(reopened_active.optimization.strategy,
                  lfs::core::param::kStrategyMCMC);
        prove_ckpt("CKPT-139");
        EXPECT_EQ(reopened_active.optimization.mask_mode,
                  expected_active.optimization.mask_mode);
        prove_ckpt("CKPT-140");
        EXPECT_TRUE(reopened_active.optimization.use_depth_loss);
        EXPECT_FLOAT_EQ(reopened_active.optimization.depth_loss_weight, 0.125f);
        prove_ckpt("CKPT-141");
        EXPECT_EQ(reopened_active.optimization.bg_mode,
                  lfs::core::param::BackgroundMode::Random);
        EXPECT_EQ(reopened_active.optimization.bg_color,
                  (std::array<float, 3>{0.125f, 0.25f, 0.5f}));
        prove_ckpt("CKPT-142");
        EXPECT_EQ(reopened_active.optimization.use_bilateral_grid,
                  expected_active.optimization.use_bilateral_grid);
        prove_ckpt("CKPT-143");
        EXPECT_EQ(reopened_active.optimization.use_ppisp,
                  expected_active.optimization.use_ppisp);
        prove_ckpt("CKPT-144");
        EXPECT_EQ(reopened_active.optimization.prune_opacity,
                  expected_active.optimization.prune_opacity);
        prove_ckpt("CKPT-145");
        EXPECT_EQ(reopened_active.optimization.use_edge_map,
                  expected_active.optimization.use_edge_map);
        prove_ckpt("CKPT-146");
        EXPECT_EQ(reopened_active.optimization.enable_sparsity,
                  expected_active.optimization.enable_sparsity);
        prove_ckpt("CKPT-147");
        EXPECT_EQ(reopened_active.dataset.images, "images_matrix_checkpoint");
        EXPECT_EQ(reopened_active.dataset.timelapse_images,
                  (std::vector<std::string>{"checkpoint_a.png", "checkpoint_b.png"}));
        EXPECT_TRUE(reopened_active.dataset.loading_params.use_16bit_color);
        prove_ckpt("CKPT-148");
        EXPECT_EQ(reopened_active.add_splat_paths, expected_active.add_splat_paths);
        const auto& frozen_ranges = target_strategy.get_model().frozen_ranges();
        ASSERT_EQ(frozen_ranges.size(), 1u);
        EXPECT_EQ(frozen_ranges[0].start, 1u);
        EXPECT_EQ(frozen_ranges[0].count, 2u);
        prove_ckpt("CKPT-149");

        const auto registered_ckpt_rows = lfs::test::licht::word_set(
            lfs::test::licht::P4_CKPT_MATRIX_ROW_DATA);
        EXPECT_EQ(proven_ckpt, registered_ckpt_rows)
            << "Every registered CKPT row must reach an explicit "
               "save->load->save assertion";

        const MatrixRows matrix_rows = read_matrix_rows();
        const auto registered = lfs::test::licht::word_set(
            lfs::test::licht::P3_MATRIX_ROW_DATA);
        EXPECT_EQ(matrix_rows.p3, registered)
            << "P3_MATRIX_ROWS must be updated whenever the normative "
               "ownership matrix gains or loses a P3 table row";
        EXPECT_EQ(proven, registered)
            << "Every registered P3 row must reach an explicit assertion "
               "after save->load->save";

        auto registered_p4 = lfs::test::licht::word_set(
            lfs::test::licht::P4_CKPT_MATRIX_ROW_DATA);
        const auto ppisp_rows = lfs::test::licht::word_set(
            lfs::test::licht::P4_PPIS_MATRIX_ROW_DATA);
        registered_p4.insert(ppisp_rows.begin(), ppisp_rows.end());
        EXPECT_EQ(matrix_rows.p4, registered_p4)
            << "P4 matrix registries must change whenever the normative "
               "ownership matrix gains or loses a CKPT/PPIS row";

        const auto registered_p5 = lfs::test::licht::word_set(
            lfs::test::licht::P5_MATRIX_ROW_DATA);
        EXPECT_EQ(matrix_rows.p5, registered_p5)
            << "P5_MATRIX_ROWS must change whenever the normative "
               "ownership matrix gains or loses a P5 row";

        // Gap #12 is normative prose rather than a Markdown table row. These
        // assertions are its machine-checkable coverage entries.
        EXPECT_EQ(point_vendor->encoding, 77u)
            << "PCLD-GAP12-242";
        EXPECT_EQ(mesh_vendor->encoding, 91u)
            << "MESH-GAP12-242";
    }

    TEST(P4MatrixProof,
         StandalonePpispSurvivesAsOneCleanLazyAuthority) {
        TemporaryDirectory temporary{"lfs-p3-matrix"};
        const auto sidecar_path =
            temporary.path / "source.ppisp";
        const auto project_path =
            temporary.path / "ppisp-proof.licht";
        const auto extracted_path =
            temporary.path / "extracted.ppisp";
        const Uuid project_uuid = matrix_uuid(500);
        const Uuid ppisp_uuid = matrix_uuid(501);

        lfs::training::PPISP source(1'000);
        source.register_frame(101, 10);
        source.register_frame(102, 20);
        source.register_frame(103, 20);
        source.finalize();
        lfs::training::PPISPControllerPool
            source_controller(2, 1'000);
        const lfs::training::PPISPFileMetadata metadata{
            .dataset_path_utf8 =
                "/non-authoritative/source",
            .images_folder = "images_4",
            .frame_image_names =
                {"a.png", "b.png", "c.png"},
            .frame_camera_ids = {10, 20, 20},
            .camera_ids = {10, 20},
        };
        const auto saved_sidecar =
            lfs::test::write_ppisp_fixture(
                sidecar_path, source,
                &source_controller, &metadata);
        ASSERT_TRUE(saved_sidecar)
            << saved_sidecar.error();

        const auto expected_bytes = lfs::test::licht::read_file_bytes(sidecar_path);
        ASSERT_FALSE(expected_bytes.empty());

        auto document = lfs::test::licht::make_empty_document(project_uuid, 500);
        auto lazy = require_result(LazyChunkValue::from_owned(
            std::make_shared<const std::vector<std::byte>>(expected_bytes), ppisp_uuid));
        require_ok(document->set_ppisp(ppisp_uuid, std::move(lazy)));
        (void)require_result(document->save(project_path, matrix_save_options(510, 600)));

        auto opened = require_result_ptr(ProjectDocument::open(project_path));
        EXPECT_TRUE(opened->checkpoint_uuids().empty());
        ASSERT_EQ(opened->ppisp_uuids().size(), 1u);
        const auto* ppisp =
            opened->find_ppisp(ppisp_uuid);
        ASSERT_NE(ppisp, nullptr);
        EXPECT_TRUE(ppisp->is_clean_reference());
        EXPECT_EQ(copy_lazy_bytes(*ppisp, 11), expected_bytes);

        auto second_save = require_result_ptr(
            opened->save(project_path, matrix_save_options(520, 700)));
        EXPECT_EQ(second_save->rewritten_chunks, 1u);
        EXPECT_EQ(second_save->reused_chunks, 10u);

        auto reopened = require_result_ptr(ProjectDocument::open(project_path));
        const auto* reopened_ppisp = reopened->find_ppisp(ppisp_uuid);
        ASSERT_NE(reopened_ppisp, nullptr);
        const auto embedded_bytes =
            copy_lazy_bytes(*reopened_ppisp, 13);
        EXPECT_EQ(embedded_bytes, expected_bytes);
        lfs::test::licht::write_file_bytes(extracted_path, embedded_bytes);

        lfs::training::PPISP loaded(1);
        lfs::training::PPISPControllerPool loaded_controller(2, 1);
        lfs::training::PPISPFileMetadata loaded_metadata;
        const auto loaded_sidecar = lfs::training::load_ppisp_file(
            extracted_path, loaded, &loaded_controller, &loaded_metadata);
        ASSERT_TRUE(loaded_sidecar) << loaded_sidecar.error();

        std::set<std::string> proven;
        const auto prove = [&](const std::string_view row) {
            EXPECT_TRUE(proven.emplace(row).second) << row;
        };
        struct Header {
            std::uint32_t magic;
            std::uint32_t version;
            std::uint32_t num_cameras;
            std::uint32_t num_frames;
            std::uint32_t flags;
            std::uint32_t reserved[3];
        };
        static_assert(sizeof(Header) == 32);
        Header header{};
        std::memcpy(&header, embedded_bytes.data(), sizeof(header));
        EXPECT_EQ(header.magic, lfs::training::PPISP_FILE_MAGIC);
        EXPECT_EQ(header.version, lfs::training::PPISP_FILE_VERSION);
        EXPECT_EQ(header.num_cameras, 2u);
        EXPECT_EQ(header.num_frames, 3u);
        prove("PPIS-157");
        EXPECT_EQ(loaded.num_cameras(), 2);
        EXPECT_EQ(loaded.num_frames(), 3);
        prove("PPIS-158");
        EXPECT_EQ(loaded_controller.num_cameras(), 2);
        EXPECT_NE(header.flags & 1u, 0u);
        prove("PPIS-159");
        EXPECT_EQ(loaded_metadata.images_folder, metadata.images_folder);
        EXPECT_EQ(loaded_metadata.frame_image_names, metadata.frame_image_names);
        EXPECT_EQ(loaded_metadata.frame_camera_ids, metadata.frame_camera_ids);
        EXPECT_EQ(loaded_metadata.camera_ids, metadata.camera_ids);
        prove("PPIS-160");

        const auto registered = lfs::test::licht::word_set(
            lfs::test::licht::P4_PPIS_MATRIX_ROW_DATA);
        EXPECT_EQ(proven, registered)
            << "Every registered PPIS row must reach an explicit "
               "save->load->save assertion";
    }

} // namespace
