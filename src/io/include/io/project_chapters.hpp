/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/parameters.hpp"
#include "core/uuid.hpp"
#include "io/json_chapter_dom.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfs::io::project {

    inline constexpr std::uint32_t JSON_CHAPTER_SCHEMA_VERSION = 1;

    struct LFS_IO_API Hash128 {
        std::array<std::uint8_t, 16> bytes{};

        [[nodiscard]] std::string to_hex() const;
        [[nodiscard]] static std::optional<Hash128> from_hex(std::string_view text);

        friend bool operator==(const Hash128&, const Hash128&) = default;
    };

    [[nodiscard]] LFS_IO_API Hash128 xxh3_128(std::span<const std::byte> bytes);

    // Incremental XXH3-128 over sequential spans. digest() matches xxh3_128 of
    // the concatenation of every update() argument, in order.
    class LFS_IO_API Hash128Stream {
    public:
        Hash128Stream();
        Hash128Stream(Hash128Stream&&) noexcept;
        Hash128Stream& operator=(Hash128Stream&&) noexcept;
        Hash128Stream(const Hash128Stream&) = delete;
        Hash128Stream& operator=(const Hash128Stream&) = delete;
        ~Hash128Stream();

        [[nodiscard]] bool valid() const noexcept;
        [[nodiscard]] bool update(std::span<const std::byte> bytes) noexcept;
        [[nodiscard]] Hash128 digest() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    struct SemanticVersion {
        std::uint16_t major = 1;
        std::uint16_t minor = 0;
        std::uint16_t patch = 0;

        friend bool operator==(const SemanticVersion&, const SemanticVersion&) = default;
    };

    struct ProjectManifest {
        std::string application_name = "LichtFeld Studio";
        SemanticVersion application_version;
        SemanticVersion schema_version{1, 0, 0};
        SemanticVersion minimum_reader_version{1, 0, 0};
        SemanticVersion minimum_safe_writer_version{1, 0, 0};
        std::vector<std::string> required_capabilities;
        std::vector<std::string> optional_capabilities;

        friend bool operator==(const ProjectManifest&, const ProjectManifest&) = default;
    };

    enum class WorldOriginProvenance : std::uint8_t {
        None,
        CentralizeByPointCloud,
        CentralizeByCameras,
        User,
        Import,
    };

    struct ProjectGeoreference {
        std::optional<std::string> crs;
        std::array<double, 3> world_origin{};
        double world_unit_scale = 1.0;
        WorldOriginProvenance world_origin_provenance = WorldOriginProvenance::None;

        friend bool operator==(const ProjectGeoreference&, const ProjectGeoreference&) = default;
    };

    enum class LocatorBase : std::uint8_t {
        Project,
        Dataset,
        Absolute,
        SearchRoot,
    };

    struct ReferenceLocator {
        std::string preferred;
        LocatorBase base = LocatorBase::Project;
        std::optional<std::string> absolute_fallback;

        friend bool operator==(const ReferenceLocator&, const ReferenceLocator&) = default;
    };

    enum class FingerprintKind : std::uint8_t {
        File,
        Directory,
    };

    struct ReferenceFingerprint {
        FingerprintKind kind = FingerprintKind::File;
        std::uint64_t size = 0;
        std::int64_t mtime_unix_ns = 0;
        Hash128 head_xxh3;
        Hash128 tail_xxh3;
        std::optional<Hash128> full_xxh3;

        friend bool operator==(const ReferenceFingerprint&, const ReferenceFingerprint&) = default;
    };

    struct ReferenceRecord {
        lfs::core::Uuid uuid;
        std::string key;
        std::string kind;
        ReferenceLocator locator;
        ReferenceFingerprint fingerprint;
        bool unresolved = false;

        friend bool operator==(const ReferenceRecord&, const ReferenceRecord&) = default;
    };

    struct EmbeddedPayloadProvenance {
        lfs::core::Uuid uuid;
        lfs::core::Uuid node_uuid;
        std::string fourcc;
        ReferenceLocator import_locator;
        ReferenceFingerprint import_fingerprint;
        Hash128 content_xxh3_128;

        friend bool operator==(const EmbeddedPayloadProvenance&,
                               const EmbeddedPayloadProvenance&) = default;
    };

    struct EmbedDecision {
        lfs::core::Uuid uuid;
        lfs::core::Uuid node_uuid;
        std::string payload_fourcc;
        std::string decision;
        std::optional<lfs::core::Uuid> reference_uuid;
        std::string reason;

        friend bool operator==(const EmbedDecision&, const EmbedDecision&) = default;
    };

    struct ProvenanceRecord {
        lfs::core::Uuid uuid;
        std::string kind;
        std::string value;

        friend bool operator==(const ProvenanceRecord&, const ProvenanceRecord&) = default;
    };

    class LFS_IO_API ProjectChapter {
    public:
        ProjectChapter();
        explicit ProjectChapter(JsonChapterDom dom);

        [[nodiscard]] static lfs::Result<ProjectChapter> parse(std::string_view bytes);
        [[nodiscard]] static lfs::Result<ProjectChapter>
        from_bytes(std::span<const std::byte> bytes);

        [[nodiscard]] const JsonChapterDom& dom() const noexcept { return dom_; }
        [[nodiscard]] JsonChapterDom& dom() noexcept { return dom_; }
        [[nodiscard]] std::vector<std::byte> to_bytes() const { return dom_.to_bytes(); }

        [[nodiscard]] lfs::Result<ProjectManifest> manifest() const;
        [[nodiscard]] lfs::Result<void> set_manifest(const ProjectManifest& value);
        [[nodiscard]] lfs::Result<lfs::core::Uuid> project_uuid() const;
        [[nodiscard]] lfs::Result<void> set_project_uuid(const lfs::core::Uuid& value);
        [[nodiscard]] lfs::Result<std::uint64_t> created_at_unix_ns() const;
        [[nodiscard]] lfs::Result<void> set_created_at_unix_ns(std::uint64_t value);
        [[nodiscard]] lfs::Result<std::uint64_t> modified_at_unix_ns() const;
        [[nodiscard]] lfs::Result<void> set_modified_at_unix_ns(std::uint64_t value);
        [[nodiscard]] lfs::Result<std::optional<lfs::core::Uuid>> dataset_reference() const;
        [[nodiscard]] lfs::Result<void>
        set_dataset_reference(std::optional<lfs::core::Uuid> value);
        [[nodiscard]] lfs::Result<std::vector<lfs::core::Uuid>> project_lineage() const;
        [[nodiscard]] lfs::Result<void>
        set_project_lineage(std::span<const lfs::core::Uuid> value);
        [[nodiscard]] lfs::Result<ProjectGeoreference> georeference() const;
        [[nodiscard]] lfs::Result<void>
        set_georeference(const ProjectGeoreference& value);

        [[nodiscard]] lfs::Result<std::vector<EmbedDecision>> embed_decisions() const;
        [[nodiscard]] lfs::Result<void> upsert_embed_decision(const EmbedDecision& value);
        [[nodiscard]] lfs::Result<std::vector<ProvenanceRecord>> provenance() const;
        [[nodiscard]] lfs::Result<void>
        upsert_provenance(const ProvenanceRecord& value);
        [[nodiscard]] lfs::Result<std::vector<EmbeddedPayloadProvenance>>
        embedded_payload_provenance() const;
        [[nodiscard]] lfs::Result<void>
        upsert_embedded_payload_provenance(const EmbeddedPayloadProvenance& value);

    private:
        JsonChapterDom dom_;
    };

    // In-memory check result only — not serialized on the wire.
    enum class FingerprintDisposition : std::uint8_t {
        MatchFastPath,
        MatchMtimeRefreshed,
        Missing,
        ContentMismatch,
        TypeMismatch,
    };

    struct FingerprintCheck {
        FingerprintDisposition disposition = FingerprintDisposition::Missing;
        std::optional<ReferenceFingerprint> observed;
        std::string diagnostic;

        [[nodiscard]] bool matches() const noexcept {
            return disposition == FingerprintDisposition::MatchFastPath ||
                   disposition == FingerprintDisposition::MatchMtimeRefreshed;
        }
    };

    [[nodiscard]] LFS_IO_API lfs::Result<ReferenceFingerprint>
    fingerprint_path(const std::filesystem::path& path, bool include_full_hash = false);
    [[nodiscard]] LFS_IO_API lfs::Result<FingerprintCheck>
    check_fingerprint(const std::filesystem::path& path,
                      const ReferenceFingerprint& expected);

    class LFS_IO_API ReferencesChapter {
    public:
        ReferencesChapter();
        explicit ReferencesChapter(JsonChapterDom dom);

        [[nodiscard]] static lfs::Result<ReferencesChapter> parse(std::string_view bytes);
        [[nodiscard]] static lfs::Result<ReferencesChapter>
        from_bytes(std::span<const std::byte> bytes);

        [[nodiscard]] const JsonChapterDom& dom() const noexcept { return dom_; }
        [[nodiscard]] JsonChapterDom& dom() noexcept { return dom_; }
        [[nodiscard]] std::vector<std::byte> to_bytes() const { return dom_.to_bytes(); }

        [[nodiscard]] lfs::Result<std::vector<ReferenceRecord>> records() const;
        [[nodiscard]] lfs::Result<std::optional<ReferenceRecord>>
        find(const lfs::core::Uuid& uuid) const;
        [[nodiscard]] lfs::Result<void> upsert(const ReferenceRecord& record);
        [[nodiscard]] lfs::Result<bool> remove(const lfs::core::Uuid& uuid);

        // A matching size+content fingerprint with mtime-only drift updates just
        // the retained mtime field. Missing/type/size/content mismatches are
        // returned as a loud failed-precondition error and leave the row intact.
        [[nodiscard]] lfs::Result<FingerprintCheck>
        verify_and_refresh(const lfs::core::Uuid& uuid,
                           const std::filesystem::path& resolved_path);
        // accept_content_change=true is reserved for a future force-relink UI.
        [[nodiscard]] lfs::Result<void>
        relink(const lfs::core::Uuid& uuid, const ReferenceLocator& locator,
               const std::filesystem::path& resolved_path,
               bool accept_content_change = false);

    private:
        JsonChapterDom dom_;
    };

    // Path ↔ REFS helpers for production adapters. Mint fingerprints a live
    // path and upserts a row (reusing the UUID for an existing key). Resolve
    // tries preferred then absolute_fallback with fingerprint precedence
    // (size+xxh3; mtime-only drift is not a relink), then falls back to hint.
    [[nodiscard]] LFS_IO_API lfs::Result<lfs::core::Uuid>
    upsert_path_reference(
        ReferencesChapter& references,
        const std::filesystem::path& project_root,
        const std::filesystem::path& live_path,
        std::string_view key,
        std::string_view kind,
        std::optional<lfs::core::Uuid> existing_uuid = std::nullopt);

    [[nodiscard]] LFS_IO_API std::optional<std::filesystem::path>
    resolve_path_reference(
        const ReferencesChapter& references,
        const std::filesystem::path& project_root,
        const lfs::core::Uuid& uuid,
        const std::filesystem::path& hint = {});

    struct GeorefPose {
        // Quaternion is stored as w,x,y,z.
        std::array<double, 4> rotation{1.0, 0.0, 0.0, 0.0};
        std::array<double, 3> translation{};

        friend bool operator==(const GeorefPose&, const GeorefPose&) = default;
    };

    struct PayloadBinding {
        std::string fourcc;
        lfs::core::Uuid instance_uuid;
        std::optional<lfs::core::Uuid> reference_uuid;
        std::string source_kind;

        friend bool operator==(const PayloadBinding&, const PayloadBinding&) = default;
    };

    struct CropBoxRecord {
        std::array<float, 3> min{-1.0f, -1.0f, -1.0f};
        std::array<float, 3> max{1.0f, 1.0f, 1.0f};
        bool inverse = false;
        bool enabled = false;
        std::array<float, 3> color{1.0f, 1.0f, 0.0f};
        float line_width = 2.0f;

        friend bool operator==(const CropBoxRecord&, const CropBoxRecord&) = default;
    };

    struct EllipsoidRecord {
        std::array<float, 3> radii{1.0f, 1.0f, 1.0f};
        bool inverse = false;
        bool enabled = false;
        std::array<float, 3> color{1.0f, 1.0f, 0.0f};
        float line_width = 2.0f;

        friend bool operator==(const EllipsoidRecord&, const EllipsoidRecord&) = default;
    };

    struct CameraRecord {
        std::int32_t uid = -1;
        std::int32_t camera_id = 0;
        std::array<float, 9> rotation{};
        std::array<float, 3> translation{};
        float focal_x = 0.0f;
        float focal_y = 0.0f;
        float center_x = 0.0f;
        float center_y = 0.0f;
        std::vector<float> radial_distortion;
        std::vector<float> tangential_distortion;
        std::int32_t camera_model_type = 0;
        std::int32_t camera_width = 0;
        std::int32_t camera_height = 0;
        std::int32_t image_width = 0;
        std::int32_t image_height = 0;
        std::string image_name;
        std::string image_path;
        std::string mask_path;
        std::string depth_path;
        std::string normal_path;
        bool has_alpha = false;
        // Load-time image presence (#1713). Independent of the node-level
        // training_enabled flag (user disable, #1716). Absent in older SCNG
        // camera objects; parsers default this to true.
        bool has_image = true;
        std::string split = "train";

        friend bool operator==(const CameraRecord&, const CameraRecord&) = default;
    };

    struct SceneNodeRecord {
        lfs::core::Uuid uuid;
        std::string type;
        std::string name;
        std::optional<lfs::core::Uuid> parent_uuid;
        std::uint32_t child_order = 0;
        // Column-major, matching glm::mat4 storage.
        std::array<float, 16> local_transform{
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1};
        bool visible = true;
        bool locked = false;
        bool training_enabled = true;
        bool payload_diverged = false;
        std::optional<GeorefPose> georef_pose;
        std::optional<PayloadBinding> payload;
        std::optional<CropBoxRecord> cropbox;
        std::optional<EllipsoidRecord> ellipsoid;
        std::optional<CameraRecord> camera;

        friend bool operator==(const SceneNodeRecord&, const SceneNodeRecord&) = default;
    };

    class LFS_IO_API SceneGraphChapter {
    public:
        SceneGraphChapter();
        explicit SceneGraphChapter(JsonChapterDom dom);

        [[nodiscard]] static lfs::Result<SceneGraphChapter> parse(std::string_view bytes);
        [[nodiscard]] static lfs::Result<SceneGraphChapter>
        from_bytes(std::span<const std::byte> bytes);

        [[nodiscard]] const JsonChapterDom& dom() const noexcept { return dom_; }
        [[nodiscard]] JsonChapterDom& dom() noexcept {
            invalidate_parsed_nodes();
            return dom_;
        }
        [[nodiscard]] std::vector<std::byte> to_bytes() const { return dom_.to_bytes(); }

        [[nodiscard]] lfs::Result<std::optional<lfs::core::Uuid>>
        training_model_uuid() const;
        [[nodiscard]] lfs::Result<void>
        set_training_model_uuid(std::optional<lfs::core::Uuid> value);
        [[nodiscard]] lfs::Result<std::vector<SceneNodeRecord>> nodes() const;
        [[nodiscard]] lfs::Result<std::optional<SceneNodeRecord>>
        find(const lfs::core::Uuid& uuid) const;
        [[nodiscard]] lfs::Result<void> upsert_node(const SceneNodeRecord& value);
        [[nodiscard]] lfs::Result<bool> remove_node(const lfs::core::Uuid& uuid);
        [[nodiscard]] lfs::Result<void> validate_hierarchy() const;
        [[nodiscard]] lfs::Result<void>
        validate_hierarchy(std::span<const SceneNodeRecord> nodes) const;

    private:
        void invalidate_parsed_nodes() noexcept;

        JsonChapterDom dom_;
        mutable std::optional<std::vector<SceneNodeRecord>> cached_nodes_;
        mutable bool hierarchy_valid_ = false;
    };

    struct ParameterManagerSnapshot {
        // Role-qualified logical identities. Raw filesystem paths are never
        // serialized in PRMS; the app-level restore adapter resolves these
        // against REFS/PPIS before presenting a path to ParameterManager.
        struct ReferenceBindings {
            std::optional<lfs::core::Uuid> background_image_reference;
            std::optional<lfs::core::Uuid> ppisp_reference;

            friend bool operator==(const ReferenceBindings&,
                                   const ReferenceBindings&) = default;
        };

        std::string active_strategy = std::string(lfs::core::param::kStrategyMRNF);
        lfs::core::param::OptimizationParameters mcmc_session;
        lfs::core::param::OptimizationParameters mrnf_session;
        lfs::core::param::OptimizationParameters igs_session;
        lfs::core::param::OptimizationParameters mcmc_current;
        lfs::core::param::OptimizationParameters mrnf_current;
        lfs::core::param::OptimizationParameters igs_current;
        ReferenceBindings mcmc_session_references;
        ReferenceBindings mrnf_session_references;
        ReferenceBindings igs_session_references;
        ReferenceBindings mcmc_current_references;
        ReferenceBindings mrnf_current_references;
        ReferenceBindings igs_current_references;
        lfs::core::param::DatasetConfig dataset;
    };

    class LFS_IO_API ParametersChapter {
    public:
        ParametersChapter();
        explicit ParametersChapter(JsonChapterDom dom);

        [[nodiscard]] static lfs::Result<ParametersChapter> parse(std::string_view bytes);
        [[nodiscard]] static lfs::Result<ParametersChapter>
        from_bytes(std::span<const std::byte> bytes);

        [[nodiscard]] const JsonChapterDom& dom() const noexcept { return dom_; }
        [[nodiscard]] JsonChapterDom& dom() noexcept {
            cached_snapshot_.reset();
            return dom_;
        }
        [[nodiscard]] std::vector<std::byte> to_bytes() const { return dom_.to_bytes(); }

        [[nodiscard]] lfs::Result<ParameterManagerSnapshot> snapshot() const;
        [[nodiscard]] lfs::Result<void>
        set_snapshot(const ParameterManagerSnapshot& value);

    private:
        JsonChapterDom dom_;
        mutable std::optional<ParameterManagerSnapshot> cached_snapshot_;
    };

    struct ReferenceOwnerBinding {
        lfs::core::Uuid reference_uuid;
        std::string chapter;
        std::optional<lfs::core::Uuid> owner_uuid;
        std::string field;

        friend bool operator==(const ReferenceOwnerBinding&,
                               const ReferenceOwnerBinding&) = default;
    };

    using ReverseReferenceIndex =
        std::unordered_map<lfs::core::Uuid, std::vector<ReferenceOwnerBinding>>;

    // additional_bindings is the adapter surface for VIEW/SEQR/CKPT chapters
    // that already expose logical reference UUIDs in later phases.
    [[nodiscard]] LFS_IO_API lfs::Result<ReverseReferenceIndex>
    build_reverse_reference_index(
        const ReferencesChapter& references, const ProjectChapter& project,
        const SceneGraphChapter& scene,
        std::span<const ReferenceOwnerBinding> additional_bindings = {});
    [[nodiscard]] LFS_IO_API lfs::Result<ReverseReferenceIndex>
    build_reverse_reference_index(
        const ReferencesChapter& references, const ProjectChapter& project,
        const SceneGraphChapter& scene, const ParametersChapter& parameters,
        std::span<const ReferenceOwnerBinding> additional_bindings = {});
    [[nodiscard]] LFS_IO_API lfs::Result<ReverseReferenceIndex>
    build_reverse_reference_index(
        std::span<const ReferenceRecord> records, const ProjectChapter& project,
        std::span<const SceneNodeRecord> nodes,
        std::span<const ReferenceOwnerBinding> additional_bindings = {});
    [[nodiscard]] LFS_IO_API lfs::Result<ReverseReferenceIndex>
    build_reverse_reference_index(
        std::span<const ReferenceRecord> records, const ProjectChapter& project,
        std::span<const SceneNodeRecord> nodes,
        const ParameterManagerSnapshot& parameters,
        std::span<const ReferenceOwnerBinding> additional_bindings = {});

} // namespace lfs::io::project
