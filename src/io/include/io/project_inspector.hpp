/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "io/project_chapters.hpp"
#include "io/project_container.hpp"
#include "io/project_recovery.hpp"
#include "io/session_chapters.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lfs::io::project {

    struct LFS_IO_API ProjectInspectorCard {
        std::filesystem::path path;
        lfs::core::Uuid project_uuid;
        lfs::core::Uuid file_uuid;
        lfs::core::Uuid commit_uuid;
        std::uint64_t generation = 0;
        std::uint64_t created_at_unix_ns = 0;
        std::uint64_t saved_at_unix_ns = 0;
        std::uint64_t physical_file_size = 0;
        ContainerRole role = ContainerRole::Master;
        OpenState open_state = OpenState::HardFail;
        std::string validation_scope = "head";
        bool has_preview = false;
        std::uint64_t preview_bytes = 0;
        std::uint32_t preview_width = 0;
        std::uint32_t preview_height = 0;
        std::optional<std::string> title;
        Version min_reader_version;
        Version min_safe_writer_version;
        CommitKind commit_kind = CommitKind::Explicit;
        std::string diagnostic;
    };

    [[nodiscard]] LFS_IO_API std::pair<std::uint32_t, std::uint32_t>
    project_preview_dimensions(const ProjectReader& reader);

    struct LFS_IO_API ProjectInspectorSave {
        std::uint64_t sequence = 0;
        std::uint64_t generation = 0;
        CommitKind kind = CommitKind::Explicit;
        std::uint64_t saved_at_unix_ns = 0;
        std::uint64_t bytes_added = 0;
        bool holds_checkpoint = false;
        std::optional<std::int32_t> checkpoint_iteration;
        std::optional<std::uint64_t> planned_iterations;
        std::string strategy;
        std::optional<std::uint32_t> gaussians;
        std::string operation;
        std::uint64_t source_save_generation = 0;
        std::uint64_t source_saved_at_unix_ns = 0;
        CommitKind source_save_kind = CommitKind::Explicit;
    };

    struct LFS_IO_API ProjectInspectorChapter {
        Fourcc fourcc;
        lfs::core::Uuid instance_uuid;
        RowKind row_kind = RowKind::Live;
        Compression compression = Compression::Stored;
        std::uint64_t stored_bytes = 0;
        std::uint64_t uncompressed_bytes = 0;
        std::uint64_t source_generation = 0;
    };

    struct LFS_IO_API ProjectInspectorCheckpoint {
        lfs::core::Uuid instance_uuid;
        std::uint64_t source_generation = 0;
        std::int32_t iteration = 0;
        std::uint32_t gaussians = 0;
        std::int32_t sh_degree = 0;
        bool binds_scene_graph = false;
        bool header_reachable = false;
        bool retained = true;
    };

    struct LFS_IO_API ProjectInspectorSceneGraph {
        std::map<std::string, std::uint64_t> node_counts_by_type;
        std::string dataset_node_name;
        std::optional<lfs::core::Uuid> training_node_id;
    };

    struct LFS_IO_API ProjectInspectorParameters {
        std::string active_strategy;
        std::optional<std::uint64_t> planned_iterations;
        bool embedded_dataset_present = false;
        bool embedded_dataset_complete = false;
        std::uint64_t embedded_images = 0;
        std::uint64_t embedded_normals = 0;
        std::uint64_t embedded_sparse = 0;
    };

    struct LFS_IO_API ProjectInspectorReference {
        std::string key;
        std::string kind;
        std::filesystem::path path;
        bool reachable = false;
    };

    struct LFS_IO_API ProjectInspectorMetrics {
        std::uint64_t loss_samples = 0;
        std::uint64_t psnr_samples = 0;
        std::optional<MetricHistorySample> last_loss;
        std::optional<MetricHistorySample> last_psnr;
        std::optional<LastEvaluationMetrics> last_evaluation;
    };

    struct LFS_IO_API ProjectInspectorDetails {
        ProjectInspectorCard card;
        ProjectStorageStats storage;
        std::vector<ProjectInspectorSave> save_history;
        std::vector<ProjectInspectorChapter> chapters;
        std::map<std::string, std::string> manifest;
        std::optional<ProjectLicense> license;
        ProjectInspectorSceneGraph scene_graph;
        ProjectInspectorParameters parameters;
        std::vector<ProjectInspectorCheckpoint> retained_checkpoints;
        std::vector<ProjectInspectorReference> references;
        ProjectInspectorMetrics metrics;
        bool autosave_sidecar_present = false;
        std::vector<std::string> chapters_requiring_full_read;
    };

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    inspect_project_card(const std::filesystem::path& path);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorDetails>
    inspect_project_details(const std::filesystem::path& path,
                            std::uint64_t checkpoint_byte_budget = 8ull * 1024 * 1024);

    [[nodiscard]] LFS_IO_API lfs::Result<std::vector<std::byte>>
    read_project_preview(const std::filesystem::path& path);

} // namespace lfs::io::project
