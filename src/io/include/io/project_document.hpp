/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "io/geometry_payload.hpp"
#include "io/project_chapters.hpp"
#include "io/project_container.hpp"
#include "io/scene_chapter_adapter.hpp"
#include "io/selection_chapter.hpp"
#include "io/splat_chapter.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace lfs::io {
    struct LoadResult;
}

namespace lfs::io::project {

    struct ProjectDocumentOpenOptions {
        ReaderOptions reader;
        GeometryDecodeOptions geometry;
    };

    struct ProjectDocumentSaveOptions {
        CommitOptions commit;
        lfs::core::Uuid file_uuid;
        IndexCompression index_compression = IndexCompression::Zstd;
        std::uint64_t disk_reserve_bytes = 64ull * 1024 * 1024;
    };

    struct ProjectDocumentSaveReport {
        std::uint64_t generation = 0;
        lfs::core::Uuid commit_uuid;
        lfs::core::Uuid snapshot_uuid;
        std::uint64_t rewritten_chunks = 0;
        std::uint64_t reused_chunks = 0;
        std::uint64_t opaque_chunks_carried = 0;
        std::uint64_t erased_chunks = 0;
    };

    struct ProjectDocumentHydrationReport {
        SelectionHydrationReport selection;
        ParameterManagerSnapshot pending_parameters;
        ReverseReferenceIndex reverse_reference_index;
    };

    class LFS_IO_API ProjectHydrationPlan {
    public:
        ProjectHydrationPlan(ProjectHydrationPlan&&) noexcept;
        ProjectHydrationPlan&
        operator=(ProjectHydrationPlan&&) noexcept;
        ProjectHydrationPlan(const ProjectHydrationPlan&) = delete;
        ProjectHydrationPlan&
        operator=(const ProjectHydrationPlan&) = delete;
        ~ProjectHydrationPlan();

        [[nodiscard]] const ProjectDocumentHydrationReport&
        report() const noexcept;

    private:
        friend class ProjectDocument;
        struct Impl;
        explicit ProjectHydrationPlan(
            std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
    };

    // A typed, retained representation of the P3 chapter set. Mutable access
    // is deliberately explicit: obtaining an edit handle marks the chapter
    // dirty. Only chapters absent from that dirty set may reuse their
    // container CleanProof.
    class LFS_IO_API ProjectDocument {
    public:
        [[nodiscard]] static lfs::Result<ProjectDocument>
        create(const lfs::core::Uuid& project_uuid,
               std::uint64_t creation_time_unix_ns = 0);
        [[nodiscard]] static lfs::Result<ProjectDocument>
        open(const std::filesystem::path& path,
             const ProjectDocumentOpenOptions& options = {});

        ProjectDocument(ProjectDocument&&) noexcept;
        ProjectDocument& operator=(ProjectDocument&&) noexcept;
        ProjectDocument(const ProjectDocument&) = delete;
        ProjectDocument& operator=(const ProjectDocument&) = delete;
        ~ProjectDocument();

        [[nodiscard]] const std::optional<std::filesystem::path>&
        source_path() const noexcept;

        [[nodiscard]] const ProjectChapter& project() const noexcept;
        [[nodiscard]] ProjectChapter& edit_project() noexcept;
        [[nodiscard]] const ReferencesChapter& references() const noexcept;
        [[nodiscard]] ReferencesChapter& edit_references() noexcept;
        [[nodiscard]] const SceneGraphChapter& scene_graph() const noexcept;
        [[nodiscard]] SceneGraphChapter& edit_scene_graph() noexcept;
        [[nodiscard]] const SelectionChapter& selection() const noexcept;
        [[nodiscard]] SelectionChapter& edit_selection() noexcept;
        [[nodiscard]] const ParametersChapter& parameters() const noexcept;
        [[nodiscard]] ParametersChapter& edit_parameters() noexcept;

        [[nodiscard]] lfs::Result<void>
        set_georeference(const ProjectGeoreference& value);
        [[nodiscard]] lfs::Result<void>
        capture_georeference(const lfs::io::LoadResult& load_result);

        [[nodiscard]] const SplatChapterPayload*
        find_splat(const lfs::core::Uuid& node_uuid) const noexcept;
        [[nodiscard]] SplatChapterPayload*
        edit_splat(const lfs::core::Uuid& node_uuid) noexcept;
        [[nodiscard]] lfs::Result<void>
        set_splat(const lfs::core::Uuid& node_uuid,
                  SplatChapterPayload payload);
        [[nodiscard]] bool remove_splat(const lfs::core::Uuid& node_uuid);

        [[nodiscard]] const PointCloudPayload*
        find_point_cloud(const lfs::core::Uuid& node_uuid) const noexcept;
        [[nodiscard]] PointCloudPayload*
        edit_point_cloud(const lfs::core::Uuid& node_uuid) noexcept;
        [[nodiscard]] lfs::Result<void>
        set_point_cloud(const lfs::core::Uuid& node_uuid,
                        PointCloudPayload payload);
        [[nodiscard]] bool remove_point_cloud(
            const lfs::core::Uuid& node_uuid);

        [[nodiscard]] const MeshPayload*
        find_mesh(const lfs::core::Uuid& node_uuid) const noexcept;
        [[nodiscard]] MeshPayload*
        edit_mesh(const lfs::core::Uuid& node_uuid) noexcept;
        [[nodiscard]] lfs::Result<void>
        set_mesh(const lfs::core::Uuid& node_uuid, MeshPayload payload);
        [[nodiscard]] bool remove_mesh(const lfs::core::Uuid& node_uuid);

        [[nodiscard]] std::vector<lfs::core::Uuid> splat_uuids() const;
        [[nodiscard]] std::vector<lfs::core::Uuid> point_cloud_uuids() const;
        [[nodiscard]] std::vector<lfs::core::Uuid> mesh_uuids() const;

        [[nodiscard]] lfs::Result<ProjectDocumentSaveReport>
        save(const std::filesystem::path& path,
             const ProjectDocumentSaveOptions& options = {});

        // Strict Phase A. Every open P3 chapter, payload, node, and selection
        // tensor is decoded and validated into a complete replacement scene.
        // destination remains untouched.
        [[nodiscard]] lfs::Result<ProjectHydrationPlan>
        stage_hydration(
            lfs::core::Scene& destination,
            const ScenePayloadResolver& external_payloads = {},
            lfs::core::SplatTensorAllocator splat_allocator = {}) const;

        // Strict Phase B. This performs only assert-guarded moves/swaps and
        // cannot parse, perform IO, or allocate.
        [[nodiscard]] static ProjectDocumentHydrationReport
        commit_hydration(
            lfs::core::Scene& destination,
            ProjectHydrationPlan&& staged) noexcept;

        // Convenience wrapper around the two explicit phases. PRMS remains
        // pending and is never applied to a running trainer.
        [[nodiscard]] lfs::Result<ProjectDocumentHydrationReport>
        hydrate(lfs::core::Scene& scene,
                const ScenePayloadResolver& external_payloads = {},
                lfs::core::SplatTensorAllocator splat_allocator = {}) const;

        [[nodiscard]] lfs::Result<ReverseReferenceIndex>
        reverse_reference_index(
            std::span<const ReferenceOwnerBinding> additional_bindings = {}) const;

    private:
        struct Impl;
        explicit ProjectDocument(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::io::project
