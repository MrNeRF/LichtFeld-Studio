/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "io/project_inspector.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace lfs::io::project {

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    undo_contents_removal(const std::filesystem::path& path, const std::string& id);

    using ProjectOperationProgress =
        std::function<void(float progress, const std::string& stage)>;
    using ProjectOperationCancel = std::function<bool()>;

    [[nodiscard]] LFS_IO_API lfs::Result<void> run_project_operation(
        const std::filesystem::path& path, const lfs::core::Uuid& expected_project,
        const lfs::core::Uuid& expected_commit, const std::function<void()>& operation);

    [[nodiscard]] LFS_IO_API lfs::Result<std::filesystem::path>
    backup_project_file(const std::filesystem::path& path);

    [[nodiscard]] LFS_IO_API lfs::Result<void> restore_project_backup(
        const std::filesystem::path& path, const std::filesystem::path& backup,
        const lfs::core::Uuid& expected_project, const lfs::core::Uuid& expected_commit);

    enum class ProjectVerificationStatus {
        Verified,
        Canceled,
        Failed,
    };

    struct LFS_IO_API ProjectVerificationResult {
        ProjectVerificationStatus status = ProjectVerificationStatus::Failed;
        std::uint64_t verified_chunks = 0;
        std::optional<std::string> first_mismatch;
    };

    struct LFS_IO_API ProjectReduceCheckpoint {
        lfs::core::Uuid instance_uuid;
        std::int32_t iteration = 0;
        std::uint64_t bytes = 0;
        bool scng_bound = false;
    };

    struct LFS_IO_API ProjectReduceDatasetPayload {
        lfs::core::Uuid chunk_uuid;
        std::string rel_path;
        std::string kind;
        std::uint64_t bytes = 0;
        bool external_replacement_validated = false;
    };

    struct LFS_IO_API ProjectReduceProjection {
        bool enabled = false;
        bool allowed = false;
        std::uint64_t reclaimable_bytes = 0;
        std::uint64_t projected_size = 0;
    };

    struct LFS_IO_API ProjectReducePlan {
        std::filesystem::path path;
        lfs::core::Uuid input_commit_uuid;
        std::uint64_t physical_size = 0;
        std::uint64_t tombstone_bytes = 0;
        std::uint64_t superseded_rows = 0;
        std::vector<ProjectReduceCheckpoint> retained_checkpoints;
        std::vector<ProjectReduceDatasetPayload> embedded_dataset;
        ProjectReduceProjection drop_checkpoints;
        ProjectReduceProjection drop_embedded_dataset;
        ProjectReduceProjection compact;
    };

    struct LFS_IO_API ProjectReduceResult {
        ProjectInspectorCard card;
        std::uint64_t checkpoints_removed = 0;
        std::uint64_t dataset_images_removed = 0;
        std::uint64_t dataset_normals_removed = 0;
        std::uint64_t dataset_sparse_removed = 0;
        std::uint64_t bytes_reclaimed = 0;
        std::filesystem::path recovery_copy;
    };

    // Explicit Contents actions append removals. Legacy reduction still compacts.
    struct ProjectReduceSelection {
        bool compact = true;
        std::optional<lfs::core::Uuid> checkpoint;
        std::uint64_t save_generation = 0;
        bool thumbnail = false;
        bool metrics = false;
    };

    struct LFS_IO_API DatasetEmbedResult {
        ProjectInspectorCard card;
        std::uint64_t images_embedded = 0;
        std::uint64_t normals_embedded = 0;
        std::uint64_t sparse_embedded = 0;
        std::uint64_t bytes_embedded = 0;
    };

    struct LFS_IO_API DatasetReferenceResult {
        ProjectInspectorCard card;
        lfs::core::Uuid reference_uuid;
        bool content_replaced = false;
    };

    enum class ProjectExportFormat {
        Ply,
        Sog,
        Ssog,
        Spz,
    };

    struct LFS_IO_API ProjectExportResult {
        std::filesystem::path destination;
        ProjectExportFormat format = ProjectExportFormat::Ply;
        std::uint64_t gaussian_count = 0;
        std::uint64_t bytes_written = 0;
    };

    struct LFS_IO_API ProjectRepairResult {
        ProjectInspectorCard card;
        std::uint64_t saves_recovered = 0;
    };

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    restore_save(const std::filesystem::path& path,
                 std::uint64_t generation,
                 const std::filesystem::path& destination);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    rebind_checkpoint(const std::filesystem::path& path,
                      const lfs::core::Uuid& checkpoint_instance_uuid);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    compact_project_file(const std::filesystem::path& path,
                         ProjectOperationProgress progress = {},
                         ProjectOperationCancel cancel = {});

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectReducePlan>
    plan_reduce_size(const std::filesystem::path& path);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectReduceResult>
    reduce_size(const std::filesystem::path& path,
                bool drop_unbound_checkpoints,
                bool drop_embedded_dataset,
                ProjectOperationProgress progress = {},
                ProjectOperationCancel cancel = {},
                const ProjectReduceSelection& selection = {});

    [[nodiscard]] LFS_IO_API lfs::Result<DatasetEmbedResult>
    embed_dataset_file(const std::filesystem::path& path,
                       ProjectOperationProgress progress = {},
                       ProjectOperationCancel cancel = {});

    [[nodiscard]] LFS_IO_API lfs::Result<DatasetReferenceResult>
    set_dataset_reference(const std::filesystem::path& path,
                          const std::filesystem::path& dataset_dir,
                          bool accept_content_change = false);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectExportResult>
    export_project_as(const std::filesystem::path& path,
                      ProjectExportFormat format,
                      const std::filesystem::path& destination,
                      ProjectOperationProgress progress = {},
                      ProjectOperationCancel cancel = {});

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectRepairResult>
    repair_project(const std::filesystem::path& path,
                   const std::filesystem::path& destination,
                   const lfs::core::Uuid& expected_project = {});

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectVerificationResult>
    verify_project_file(const std::filesystem::path& path,
                        ProjectOperationProgress progress = {},
                        ProjectOperationCancel cancel = {});

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    set_project_preview(const std::filesystem::path& path,
                        std::span<const std::byte> png_bytes);

    struct LFS_IO_API ProjectThumbnailSourceAvailability {
        bool first_dataset_image = false;
        bool first_embedded_image = false;
    };

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectThumbnailSourceAvailability>
    inspect_project_thumbnail_sources(const std::filesystem::path& path);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    preview_from_first_dataset_image(const std::filesystem::path& path);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    preview_from_first_embedded_image(const std::filesystem::path& path);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    preview_from_image_file(const std::filesystem::path& project_path,
                            const std::filesystem::path& image_path);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    set_project_license(const std::filesystem::path& path,
                        const std::string& identifier,
                        const std::string& notice = {});

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    clear_project_license(const std::filesystem::path& path);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    set_project_title(const std::filesystem::path& path,
                      const std::string& title);

} // namespace lfs::io::project
