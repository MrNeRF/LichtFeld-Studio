/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/checkpoint_format.hpp"
#include "core/error.hpp"
#include "core/export.hpp"
#include "io/geometry_payload.hpp"
#include "io/project_chapters.hpp"
#include "io/project_container.hpp"
#include "io/scene_chapter_adapter.hpp"
#include "io/selection_chapter.hpp"
#include "io/session_chapters.hpp"
#include "io/splat_chapter.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace lfs::io {
    struct LoadResult;
}

namespace lfs::io::project {

    // A binary chapter whose clean source range is also its lazy hydration
    // handle. Reading a clean value never marks it dirty; saving that same
    // value reuses its CleanProof byte-for-byte. An owned value retains the
    // producer's immutable storage and is streamed without a second
    // checkpoint-sized allocation.
    class LFS_IO_API LazyChunkValue {
    public:
        using StreamVisitor =
            std::function<lfs::Result<void>(std::istream&, std::uint64_t)>;

        LazyChunkValue(LazyChunkValue&&) noexcept;
        LazyChunkValue& operator=(LazyChunkValue&&) noexcept;
        LazyChunkValue(const LazyChunkValue&) = delete;
        LazyChunkValue& operator=(const LazyChunkValue&) = delete;
        ~LazyChunkValue();

        [[nodiscard]] static lfs::Result<LazyChunkValue>
        from_owned(std::shared_ptr<const std::vector<std::byte>> bytes,
                   const lfs::core::Uuid& snapshot_uuid);
        [[nodiscard]] static lfs::Result<LazyChunkValue>
        from_owned(std::vector<std::byte> bytes,
                   const lfs::core::Uuid& snapshot_uuid);

        [[nodiscard]] std::uint64_t size() const noexcept;
        [[nodiscard]] const lfs::core::Uuid& snapshot_uuid() const noexcept;
        [[nodiscard]] bool is_clean_reference() const noexcept;
        // Test-only: forget the CleanProof while keeping the file-backed
        // source so save() cannot reuse the row and must copy stored bytes.
        void drop_clean_proof_for_testing() noexcept;

        [[nodiscard]] lfs::Result<void>
        read_at(std::uint64_t offset, std::span<std::byte> destination) const;
        [[nodiscard]] lfs::Result<void>
        visit_stream(const StreamVisitor& visitor) const;

    private:
        friend class ProjectDocument;
        struct Impl;
        explicit LazyChunkValue(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
    };

    struct ProjectDocumentOpenOptions {
        ReaderOptions reader;
        GeometryDecodeOptions geometry;
        // Decode the KB-scale shell chapters only. Embedded scene payloads
        // remain clean source spans until stage_hydration() consumes them.
        bool defer_geometry_payloads = false;
    };

    struct ProjectDocumentSaveOptions {
        CommitOptions commit;
        lfs::core::Uuid file_uuid;
        // A titled-project Save As uses a new catalog identity. Leave null
        // for ordinary saves, recovery publication, and first save.
        lfs::core::Uuid save_as_project_uuid = {};
        IndexCompression index_compression = IndexCompression::Zstd;
        std::uint64_t disk_reserve_bytes = 64ull * 1024 * 1024;
        // Only a file-dialog-confirmed Save As may replace a first-save destination.
        bool allow_existing_destination_replacement = false;
        // Explicit GUI saves may replace THMB. An empty span means carry the
        // current preview forward without regenerating it.
        std::span<const std::byte> preview_png;
        // Optional deterministic seam for Save As's internal compaction
        // generation. Normal callers leave these unset.
        lfs::core::Uuid save_as_compaction_commit_uuid = {};
        std::uint64_t save_as_creation_time_unix_ns = 0;
        // Recovery merge writers reuse the original master's retained lock.
        std::optional<WriterLockLease> writer_lock_lease =
            std::nullopt;
        // Terminal training writes wait out a transient in-process lock
        // holder instead of dropping the generation.
        std::chrono::milliseconds writer_lock_wait{0};
        // Untitled crash-protection writes a complete master without binding
        // the live document to that app-private path.
        bool leave_unbound = false;
    };

    [[nodiscard]] LFS_IO_API lfs::Result<std::vector<std::byte>>
    dataset_preview_png(const std::filesystem::path& first_image,
                        int max_size = 512);

    [[nodiscard]] LFS_IO_API std::optional<std::filesystem::path>
    first_dataset_image(const lfs::core::param::DatasetConfig& dataset);

    [[nodiscard]] LFS_IO_API std::optional<std::filesystem::path>
    first_dataset_image(const ProjectChapter& project,
                        const ReferencesChapter& references,
                        const ParametersChapter& parameters,
                        const std::filesystem::path& project_root = {});

    struct ProjectDocumentAutosaveOptions {
        lfs::core::Uuid file_uuid;
        lfs::core::Uuid base_explicit_commit_uuid;
        std::uint64_t autosave_sequence = 0;
        lfs::core::Uuid snapshot_uuid;
        // Optional deterministic release-fixture seam. Production callers
        // leave these unset and receive a generated identity/current time.
        lfs::core::Uuid commit_uuid = {};
        std::uint64_t wallclock_unix_ns = 0;
        IndexCompression index_compression = IndexCompression::Zstd;
        std::uint64_t disk_reserve_bytes = 64ull * 1024 * 1024;
        std::optional<WriterLockLease> writer_lock_lease =
            std::nullopt;
        CommitBoundaryObserver boundary_observer;
    };

    struct ProjectDocumentPayloadState {
        Fourcc fourcc;
        lfs::core::Uuid instance_uuid;
        bool loaded = true;
    };

    enum class ProjectDocumentDegradedState : std::uint8_t {
        MissingActiveCamera = 1,
        MissingPlySequenceNode = 2,
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
        std::optional<lfs::core::Uuid> checkpoint_uuid;
        std::optional<lfs::core::CheckpointHeader> checkpoint_header;
        bool trainer_state_pending = false;
        ProjectSessionChapters pending_session;
        std::size_t hydrated_payload_units = 0;
        std::size_t invalidated_payload_units = 0;
        bool selection_installed = false;
        double splat_read_ms = 0;
        double splat_hash_ms = 0;
        double splat_copy_ms = 0;
        double splat_materialize_ms = 0;
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

    // A typed, retained representation of the project chapter set. Mutable
    // access is deliberately explicit: obtaining an edit handle marks the
    // chapter dirty. Only chapters absent from that dirty set may reuse their
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
        // Keep the source reader for lazy payloads, but report no user path.
        void forget_source_path() noexcept;
        [[nodiscard]] const ProjectReader* source_reader() const noexcept;
        [[nodiscard]] std::optional<lfs::core::Uuid>
        source_commit_uuid() const noexcept;
        [[nodiscard]] std::span<const ProjectDocumentDegradedState>
        degraded_states() const noexcept;
        [[nodiscard]] const lfs::core::Uuid& project_uuid() const noexcept;
        [[nodiscard]] std::uint64_t generation() const noexcept;
        [[nodiscard]] std::uint64_t dirty_epoch() const noexcept;
        [[nodiscard]] bool dirty() const noexcept;
        [[nodiscard]] std::vector<std::string> dirty_chapters() const;
        [[nodiscard]] std::vector<ProjectDocumentPayloadState>
        payload_states() const;

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
        [[nodiscard]] const GuiLayoutChapter& gui_layout() const noexcept;
        [[nodiscard]] GuiLayoutChapter& edit_gui_layout() noexcept;
        [[nodiscard]] const ViewSessionChapter& view() const noexcept;
        [[nodiscard]] ViewSessionChapter& edit_view() noexcept;
        [[nodiscard]] const EditorSessionChapter& editor() const noexcept;
        [[nodiscard]] EditorSessionChapter& edit_editor() noexcept;
        [[nodiscard]] const SequencerSessionChapter& sequencer() const noexcept;
        [[nodiscard]] SequencerSessionChapter& edit_sequencer() noexcept;
        [[nodiscard]] const MetricsChapter& metrics() const noexcept;
        [[nodiscard]] MetricsChapter& edit_metrics() noexcept;

        [[nodiscard]] const LazyChunkValue*
        find_checkpoint(const lfs::core::Uuid& instance_uuid) const noexcept;
        // Test-only: drop the file-backed CKPT CleanProof. find_checkpoint()
        // is const and LazyChunkValue::Impl is translation-unit local.
        void drop_checkpoint_clean_proof_for_testing(
            const lfs::core::Uuid& instance_uuid);
        [[nodiscard]] lfs::Result<void>
        set_checkpoint(const lfs::core::Uuid& instance_uuid,
                       LazyChunkValue payload);
        [[nodiscard]] bool
        remove_checkpoint(const lfs::core::Uuid& instance_uuid);
        [[nodiscard]] std::vector<lfs::core::Uuid>
        checkpoint_uuids() const;

        [[nodiscard]] const LazyChunkValue*
        find_ppisp(const lfs::core::Uuid& instance_uuid) const noexcept;
        [[nodiscard]] lfs::Result<void>
        set_ppisp(const lfs::core::Uuid& instance_uuid,
                  LazyChunkValue payload);
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
        // Drop SPLT/PCLD/MESH payloads the current scene graph no longer binds.
        void remove_geometry_payloads_not_bound_by_scene();

        [[nodiscard]] std::vector<lfs::core::Uuid> splat_uuids() const;
        [[nodiscard]] std::vector<lfs::core::Uuid> point_cloud_uuids() const;
        [[nodiscard]] std::vector<lfs::core::Uuid> mesh_uuids() const;

        [[nodiscard]] lfs::Result<ProjectDocumentSaveReport>
        save(const std::filesystem::path& path,
             const ProjectDocumentSaveOptions& options = {});
        // Publishes a compacted sibling with a new file UUID and all
        // clean/unloaded payloads. save_as_project_uuid optionally assigns a
        // new project identity. An existing destination is atomically
        // replaced only after staged verification.
        [[nodiscard]] lfs::Result<ProjectDocumentSaveReport>
        save_as(const std::filesystem::path& path,
                const ProjectDocumentSaveOptions& options = {});
        // Builds one complete overlay relative to the currently opened
        // master. It never clears dirty state or regenerates THMB.
        [[nodiscard]] lfs::Result<ProjectDocumentSaveReport>
        save_autosave(
            const std::filesystem::path& sidecar_path,
            const ProjectDocumentAutosaveOptions& options);

        // Phase-A interactive shell. Heavy geometry and selection masks stay
        // deferred, while nodes and selection-group metadata are coherent.
        [[nodiscard]] lfs::Result<std::unique_ptr<lfs::core::Scene>>
        stage_shell(lfs::core::Scene& destination) const;

        // Strict Phase A. Every project chapter, payload, node, selection
        // tensor, and five-chapter GUI session bundle is decoded and validated
        // before destination is touched.
        [[nodiscard]] lfs::Result<ProjectHydrationPlan>
        stage_hydration(
            lfs::core::Scene& destination,
            const ScenePayloadResolver& external_payloads = {},
            lfs::core::SplatTensorAllocator splat_allocator = {},
            std::function<void(std::size_t, std::size_t)> payload_progress = {}) const;

        // Strict Phase B. This performs only assert-guarded moves/swaps and
        // cannot parse, perform IO, or allocate.
        [[nodiscard]] static ProjectDocumentHydrationReport
        commit_hydration(
            lfs::core::Scene& destination,
            ProjectHydrationPlan&& staged) noexcept;
        // Phase B for an already interactive shell. Payloads are attached at
        // the node/chapter boundary; live edits invalidate only their UUID.
        [[nodiscard]] static ProjectDocumentHydrationReport
        commit_partial_hydration(
            lfs::core::Scene& destination,
            ProjectHydrationPlan&& staged,
            bool install_selection) noexcept;

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
        [[nodiscard]] lfs::Result<ProjectDocumentSaveReport>
        save_impl(
            const std::filesystem::path& path,
            const ProjectDocumentSaveOptions& options,
            const ProjectDocumentAutosaveOptions* autosave);
        struct Impl;
        explicit ProjectDocument(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::io::project
