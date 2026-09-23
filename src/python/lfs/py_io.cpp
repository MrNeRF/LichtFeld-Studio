/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_io.hpp"
#include "py_cameras.hpp"
#include "py_error.hpp"
#include "py_scene.hpp"
#include "py_splat_data.hpp"
#include "py_tensor.hpp"

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/provenance.hpp"
#include "core/splat_data.hpp"
#include "core/user_paths.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"
#include "io/ply_export_internal.hpp"
#include "io/project_chapters.hpp"
#include "io/project_container.hpp"
#include "io/project_document.hpp"
#include "io/project_inspector.hpp"
#include "io/project_operations.hpp"
#include "io/project_recovery.hpp"
#include "io/splat_path.hpp"
#include "training/dataset.hpp"

#include <filesystem>
#include <format>
#include <optional>
#include <span>

namespace lfs::python {

    namespace {

        // Phase 9 Section 1.5: py_io is the one reference binding group converted
        // to typed errors. Each throw carries a correct ErrorCode + ErrorDomain::IO
        // so the LIFO translator maps it to the right lichtfeld.* subclass.
        // NOTE: deliberate fork of io/error.hpp's to_lfs_error_code: this map biases
        // filesystem-shaped codes toward NotFound and header/JSON damage toward
        // InvalidArgument (Python exception ergonomics), and throw_io_error emits the
        // Phase 9 pinned attr name `requested_bytes` where io::to_lfs_error emits
        // `required_bytes`. When io::ErrorCode gains a value, update BOTH maps.
        lfs::ErrorCode map_io_code(const io::ErrorCode code) noexcept {
            switch (code) {
            case io::ErrorCode::PATH_NOT_FOUND:
            case io::ErrorCode::NOT_A_DIRECTORY:
            case io::ErrorCode::NOT_A_FILE:
            case io::ErrorCode::MISSING_REQUIRED_FILES:
                return lfs::ErrorCode::NotFound;
            case io::ErrorCode::PERMISSION_DENIED:
            case io::ErrorCode::PATH_NOT_WRITABLE:
                return lfs::ErrorCode::PermissionDenied;
            case io::ErrorCode::INSUFFICIENT_DISK_SPACE:
            case io::ErrorCode::RESOURCE_EXHAUSTED:
                return lfs::ErrorCode::ResourceExhausted;
            case io::ErrorCode::INVALID_DATASET:
            case io::ErrorCode::EMPTY_DATASET:
            case io::ErrorCode::INVALID_HEADER:
            case io::ErrorCode::MALFORMED_JSON:
            case io::ErrorCode::MASK_SIZE_MISMATCH:
            case io::ErrorCode::DEPTH_SIZE_MISMATCH:
            case io::ErrorCode::NORMAL_SIZE_MISMATCH:
                return lfs::ErrorCode::InvalidArgument;
            case io::ErrorCode::UNSUPPORTED_FORMAT:
                return lfs::ErrorCode::Unsupported;
            case io::ErrorCode::CORRUPTED_DATA:
            case io::ErrorCode::DECODING_FAILED:
                return lfs::ErrorCode::DataLoss;
            case io::ErrorCode::CANCELLED:
                return lfs::ErrorCode::Cancelled;
            case io::ErrorCode::WRITE_FAILURE:
            case io::ErrorCode::ENCODING_FAILED:
            case io::ErrorCode::ARCHIVE_CREATION_FAILED:
            case io::ErrorCode::READ_FAILURE:
            case io::ErrorCode::INTERNAL_ERROR:
            case io::ErrorCode::SUCCESS:
                return lfs::ErrorCode::Internal;
            }
            return lfs::ErrorCode::Internal;
        }

        [[noreturn]] void throw_io_error(const io::Error& error, const std::string& context,
                                         const lfs::core::SourceSite site = LFS_SOURCE_SITE_CURRENT()) {
            lfs::SmallFields fields;
            if (!error.path.empty())
                fields.add("path", lfs::core::path_to_utf8(error.path));
            if (error.required_bytes != 0)
                fields.add("requested_bytes", static_cast<std::uint64_t>(error.required_bytes));
            if (error.available_bytes != 0)
                fields.add("available_bytes", static_cast<std::uint64_t>(error.available_bytes));
            const std::string formatted = error.format();
            throw lfs::Exception(lfs::make_error({
                .code = map_io_code(error.code),
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .user_message = context.empty() ? formatted : std::format("{}: {}", context, formatted),
                .detail = formatted,
                .detection = site,
                .fields = std::move(fields),
            }));
        }

        [[noreturn]] void throw_invalid_io_argument(std::string message,
                                                    const lfs::core::SourceSite site = LFS_SOURCE_SITE_CURRENT()) {
            throw lfs::Exception(lfs::make_error({
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .user_message = std::move(message),
                .detection = site,
            }));
        }

        struct PyProgressCallback {
            nb::object callback;

            void operator()(float progress, const std::string& message) const {
                nb::gil_scoped_acquire gil;
                if (!callback || callback.is_none())
                    return;
                try {
                    callback(progress, message);
                } catch (const std::exception& e) {
                    LOG_ERROR("Python progress callback error: {}", e.what());
                }
            }
        };

        struct PyCancelCallback {
            nb::object callback;

            bool operator()() const {
                nb::gil_scoped_acquire gil;
                if (!callback || callback.is_none())
                    return false;
                try {
                    return nb::cast<bool>(callback());
                } catch (const std::exception& e) {
                    LOG_ERROR("Python cancellation callback error: {}", e.what());
                    return true;
                }
            }
        };

        struct PyExportProgressCallback {
            nb::object callback;

            bool operator()(float progress, const std::string& stage) const {
                nb::gil_scoped_acquire gil;
                if (!callback)
                    return true;
                try {
                    nb::object result = callback(progress, stage);
                    if (nb::isinstance<nb::bool_>(result))
                        return nb::cast<bool>(result);
                    return true;
                } catch (const std::exception& e) {
                    LOG_ERROR("Python export progress callback error: {}", e.what());
                    return false;
                }
            }
        };

        struct PyLoadResult {
            std::shared_ptr<core::SplatData> splat_data;
            std::vector<std::shared_ptr<core::Camera>> cameras;
            std::shared_ptr<core::PointCloud> point_cloud;
            PyTensor scene_center;
            std::string loader_used;
            int64_t load_time_ms;
            std::vector<std::string> warnings;

            std::optional<PySplatData> get_splat_data() const {
                if (splat_data)
                    return PySplatData(splat_data);
                return std::nullopt;
            }

            std::optional<PyCameraDataset> get_cameras() const {
                if (cameras.empty())
                    return std::nullopt;
                training::DatasetConfig config;
                auto dataset = std::make_shared<training::CameraDataset>(
                    cameras, config, training::CameraDataset::Split::ALL);
                return PyCameraDataset(dataset);
            }

            std::optional<PyPointCloud> get_point_cloud() const {
                if (point_cloud)
                    return PyPointCloud(point_cloud.get());
                return std::nullopt;
            }

            bool is_dataset() const { return !cameras.empty(); }
        };

        struct PyProjectInspection {
            std::string project_uuid;
            std::string file_uuid;
            std::string commit_uuid;
            std::uint64_t generation = 0;
            std::uint64_t created_at_unix_ns = 0;
            std::uint64_t saved_at_unix_ns = 0;
            std::uint64_t physical_file_size = 0;
            io::project::ContainerRole role = io::project::ContainerRole::Master;
            io::project::OpenState open_state = io::project::OpenState::HardFail;
            bool has_preview = false;
            bool has_checkpoint = false;
            bool has_dataset = false;
            std::uint32_t preview_width = 0;
            std::uint32_t preview_height = 0;
            std::string fallback_preview_path;
        };

        core::Tensor tensor_from_python_attribute(const nb::handle& value) {
            if (nb::isinstance<PyTensor>(value)) {
                return nb::cast<PyTensor>(value).tensor();
            }

            if (nb::isinstance<nb::ndarray<>>(value)) {
                return PyTensor::from_numpy(nb::cast<nb::ndarray<>>(value)).tensor();
            }

            throw_invalid_io_argument(
                "extra_attributes values must be lichtfeld.Tensor or numpy.ndarray");
        }

        std::vector<io::PlyAttributeBlock> parse_extra_ply_attributes(const nb::object& extra_attributes,
                                                                      const std::filesystem::path& output_path) {
            if (!extra_attributes || extra_attributes.is_none()) {
                return {};
            }

            if (!nb::isinstance<nb::dict>(extra_attributes)) {
                throw_invalid_io_argument(
                    "extra_attributes must be a dict[str, lichtfeld.Tensor | numpy.ndarray]");
            }

            nb::dict attributes = nb::cast<nb::dict>(extra_attributes);
            std::vector<io::PlyAttributeBlock> blocks;
            blocks.reserve(attributes.size());

            for (const auto& item : attributes) {
                const std::string name = nb::cast<std::string>(item.first);
                if (name.empty()) {
                    throw_invalid_io_argument("extra_attributes keys must not be empty");
                }

                auto values = tensor_from_python_attribute(item.second);
                if (!values.is_valid() || values.numel() == 0) {
                    throw_invalid_io_argument(std::format(
                        "extra_attributes['{}'] must not be empty", name));
                }

                if (values.ndim() != 1 && values.ndim() != 2) {
                    throw_invalid_io_argument(std::format(
                        "extra_attributes['{}'] must be shaped [N] or [N,C]", name));
                }

                const size_t cols = values.ndim() == 1 ? 1 : static_cast<size_t>(values.size(1));
                if (cols == 0) {
                    throw_invalid_io_argument(std::format(
                        "extra_attributes['{}'] must have at least one column", name));
                }

                auto names = io::make_ply_extra_attribute_names(name, cols);
                if (auto result = io::validate_reserved_ply_extra_attribute_names(names, output_path); !result) {
                    throw_io_error(result.error(), "Invalid extra PLY attribute name");
                }

                blocks.push_back(io::PlyAttributeBlock{
                    .values = std::move(values),
                    .names = std::move(names),
                });
            }

            return blocks;
        }

        core::Uuid parse_reference_uuid(const std::string& value) {
            const auto parsed = core::Uuid::from_string(value);
            if (!parsed) {
                throw_invalid_io_argument(std::format("Invalid reference UUID: {}", value));
            }
            return *parsed;
        }

    } // namespace

    void register_io(nb::module_& m) {
        namespace project = io::project;

        nb::class_<project::Hash128>(m, "Hash128")
            .def(nb::init<>())
            .def_static(
                "from_hex",
                [](const std::string& value) {
                    const auto parsed = project::Hash128::from_hex(value);
                    if (!parsed) {
                        throw_invalid_io_argument("Hash128 must contain exactly 32 hexadecimal characters");
                    }
                    return *parsed;
                },
                nb::arg("value"))
            .def("to_hex", &project::Hash128::to_hex)
            .def("__str__", &project::Hash128::to_hex);

        nb::enum_<project::LocatorBase>(m, "LocatorBase")
            .value("PROJECT", project::LocatorBase::Project)
            .value("DATASET", project::LocatorBase::Dataset)
            .value("ABSOLUTE", project::LocatorBase::Absolute)
            .value("SEARCH_ROOT", project::LocatorBase::SearchRoot);

        nb::class_<project::ReferenceLocator>(m, "ReferenceLocator")
            .def(nb::init<>())
            .def_rw("preferred", &project::ReferenceLocator::preferred)
            .def_rw("base", &project::ReferenceLocator::base)
            .def_rw("absolute_fallback", &project::ReferenceLocator::absolute_fallback);

        nb::enum_<project::FingerprintKind>(m, "FingerprintKind")
            .value("FILE", project::FingerprintKind::File)
            .value("DIRECTORY", project::FingerprintKind::Directory);

        nb::class_<project::ReferenceFingerprint>(m, "ReferenceFingerprint")
            .def(nb::init<>())
            .def_rw("kind", &project::ReferenceFingerprint::kind)
            .def_rw("size", &project::ReferenceFingerprint::size)
            .def_rw("mtime_unix_ns", &project::ReferenceFingerprint::mtime_unix_ns)
            .def_rw("head_xxh3", &project::ReferenceFingerprint::head_xxh3)
            .def_rw("tail_xxh3", &project::ReferenceFingerprint::tail_xxh3)
            .def_rw("full_xxh3", &project::ReferenceFingerprint::full_xxh3);

        nb::enum_<project::FingerprintDisposition>(m, "FingerprintDisposition")
            .value("MATCH_FAST_PATH", project::FingerprintDisposition::MatchFastPath)
            .value("MATCH_MTIME_REFRESHED", project::FingerprintDisposition::MatchMtimeRefreshed)
            .value("MISSING", project::FingerprintDisposition::Missing)
            .value("CONTENT_MISMATCH", project::FingerprintDisposition::ContentMismatch)
            .value("TYPE_MISMATCH", project::FingerprintDisposition::TypeMismatch);

        nb::class_<project::FingerprintCheck>(m, "FingerprintCheck")
            .def_prop_ro("disposition", [](const project::FingerprintCheck& value) { return value.disposition; })
            .def_prop_ro("observed", [](const project::FingerprintCheck& value) { return value.observed; })
            .def_prop_ro("diagnostic", [](const project::FingerprintCheck& value) { return value.diagnostic; })
            .def_prop_ro("matches", &project::FingerprintCheck::matches);

        nb::class_<project::ReferenceRecord>(m, "ReferenceRecord")
            .def(nb::init<>())
            .def_prop_rw(
                "uuid",
                [](const project::ReferenceRecord& value) { return value.uuid.to_string(); },
                [](project::ReferenceRecord& value, const std::string& uuid) { value.uuid = parse_reference_uuid(uuid); })
            .def_rw("key", &project::ReferenceRecord::key)
            .def_rw("kind", &project::ReferenceRecord::kind)
            .def_rw("locator", &project::ReferenceRecord::locator)
            .def_rw("fingerprint", &project::ReferenceRecord::fingerprint)
            .def_rw("unresolved", &project::ReferenceRecord::unresolved);

        nb::class_<project::ReferencesChapter>(m, "ReferencesChapter")
            .def(nb::init<>())
            .def_static(
                "parse",
                [](const std::string& value) { return unwrap(project::ReferencesChapter::parse(value)); },
                nb::arg("value"))
            .def(
                "to_json",
                [](const project::ReferencesChapter& value) {
                    const auto bytes = value.to_bytes();
                    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                })
            .def("records", [](const project::ReferencesChapter& value) { return unwrap(value.records()); })
            .def(
                "find",
                [](const project::ReferencesChapter& value, const std::string& uuid) {
                    return unwrap(value.find(parse_reference_uuid(uuid)));
                },
                nb::arg("uuid"))
            .def(
                "upsert",
                [](project::ReferencesChapter& value, const project::ReferenceRecord& record) {
                    unwrap(value.upsert(record));
                },
                nb::arg("record"))
            .def(
                "remove",
                [](project::ReferencesChapter& value, const std::string& uuid) {
                    return unwrap(value.remove(parse_reference_uuid(uuid)));
                },
                nb::arg("uuid"))
            .def(
                "verify_and_refresh",
                [](project::ReferencesChapter& value, const std::string& uuid,
                   const std::filesystem::path& path) {
                    const auto parsed = parse_reference_uuid(uuid);
                    std::optional<lfs::Result<project::FingerprintCheck>> result;
                    {
                        nb::gil_scoped_release release;
                        result = value.verify_and_refresh(parsed, path);
                    }
                    return unwrap(std::move(*result));
                },
                nb::arg("uuid"), nb::arg("path"))
            .def(
                "relink",
                [](project::ReferencesChapter& value, const std::string& uuid,
                   const project::ReferenceLocator& locator, const std::filesystem::path& path,
                   const bool accept_content_change) {
                    const auto parsed = parse_reference_uuid(uuid);
                    lfs::Status result;
                    {
                        nb::gil_scoped_release release;
                        result = value.relink(parsed, locator, path, accept_content_change);
                    }
                    unwrap(std::move(result));
                },
                nb::arg("uuid"), nb::arg("locator"), nb::arg("path"),
                nb::arg("accept_content_change") = false);

        m.def(
            "fingerprint_path",
            [](const std::filesystem::path& path, const bool include_full_hash) {
                std::optional<lfs::Result<project::ReferenceFingerprint>> result;
                {
                    nb::gil_scoped_release release;
                    result = project::fingerprint_path(path, include_full_hash);
                }
                return unwrap(std::move(*result));
            },
            nb::arg("path"), nb::arg("include_full_hash") = false,
            "Fingerprint a file or directory for durable content identity.");

        m.def(
            "check_fingerprint",
            [](const std::filesystem::path& path, const project::ReferenceFingerprint& expected) {
                std::optional<lfs::Result<project::FingerprintCheck>> result;
                {
                    nb::gil_scoped_release release;
                    result = project::check_fingerprint(path, expected);
                }
                return unwrap(std::move(*result));
            },
            nb::arg("path"), nb::arg("expected"),
            "Compare a path with a previously stored content fingerprint.");

        m.def(
            "asset_library_dir",
            []() {
                return unwrap(core::UserPaths::resolve()).assetLibraryDir();
            },
            "Return the canonical user Asset Manager storage directory.");

        nb::enum_<project::ContainerRole>(m, "ProjectContainerRole")
            .value("MASTER", project::ContainerRole::Master)
            .value("AUTOSAVE_SIDECAR", project::ContainerRole::AutosaveSidecar);

        nb::enum_<project::OpenState>(m, "ProjectOpenState")
            .value("OPEN", project::OpenState::Open)
            .value("UNSUPPORTED_NEWER", project::OpenState::UnsupportedNewer)
            .value("REPAIR_ONLY", project::OpenState::RepairOnly)
            .value("HARD_FAIL", project::OpenState::HardFail);

        nb::enum_<project::CommitKind>(m, "ProjectCommitKind")
            .value("EXPLICIT", project::CommitKind::Explicit)
            .value("AUTOSAVE", project::CommitKind::Autosave)
            .value("RECOVERED", project::CommitKind::Recovered)
            .value("COMPACTION", project::CommitKind::Compaction)
            .value("CONTENTS", project::CommitKind::Contents);

        nb::enum_<project::RowKind>(m, "ProjectRowKind")
            .value("LIVE", project::RowKind::Live)
            .value("TOMBSTONE", project::RowKind::Tombstone)
            .value("SIDECAR_BASE_REFERENCE", project::RowKind::SidecarBaseReference);

        nb::enum_<project::Compression>(m, "ProjectCompression")
            .value("STORED", project::Compression::Stored)
            .value("ZSTD_FRAMED", project::Compression::ZstdFramed)
            .value("BYTE_SHUFFLE_ZSTD_FRAMED", project::Compression::ByteShuffleZstdFramed);

        nb::class_<project::Version>(m, "ProjectVersion")
            .def(nb::init<>())
            .def_ro("major", &project::Version::major)
            .def_ro("minor", &project::Version::minor);

        nb::class_<PyProjectInspection>(m, "ProjectInspection")
            .def_ro("project_uuid", &PyProjectInspection::project_uuid)
            .def_ro("file_uuid", &PyProjectInspection::file_uuid)
            .def_ro("commit_uuid", &PyProjectInspection::commit_uuid)
            .def_ro("generation", &PyProjectInspection::generation)
            .def_ro("created_at_unix_ns", &PyProjectInspection::created_at_unix_ns)
            .def_ro("saved_at_unix_ns", &PyProjectInspection::saved_at_unix_ns)
            .def_ro("physical_file_size", &PyProjectInspection::physical_file_size)
            .def_ro("role", &PyProjectInspection::role)
            .def_ro("open_state", &PyProjectInspection::open_state)
            .def_ro("has_preview", &PyProjectInspection::has_preview)
            .def_ro("has_checkpoint", &PyProjectInspection::has_checkpoint)
            .def_ro("has_dataset", &PyProjectInspection::has_dataset)
            .def_ro("preview_width", &PyProjectInspection::preview_width)
            .def_ro("preview_height", &PyProjectInspection::preview_height)
            .def_ro("fallback_preview_path", &PyProjectInspection::fallback_preview_path);

        nb::class_<project::OpenClassification>(m, "ProjectOpenClassification")
            .def_ro("state", &project::OpenClassification::state)
            .def_ro("generation", &project::OpenClassification::generation)
            .def_ro("diagnostic", &project::OpenClassification::diagnostic);

        nb::class_<project::ProjectStorageStats>(m, "ProjectStorageStats")
            .def_ro("physical_bytes", &project::ProjectStorageStats::physical_bytes)
            .def_ro("estimated_live_bytes", &project::ProjectStorageStats::estimated_live_bytes)
            .def_ro("dead_bytes", &project::ProjectStorageStats::dead_bytes)
            .def_ro("dead_ratio", &project::ProjectStorageStats::dead_ratio);

        nb::class_<project::ProjectLicense>(m, "ProjectLicense")
            .def(nb::init<>())
            .def_rw("identifier", &project::ProjectLicense::identifier)
            .def_rw("notice", &project::ProjectLicense::notice);

        nb::class_<project::ProjectInspectorCard>(m, "ProjectInspectorCard")
            .def_ro("path", &project::ProjectInspectorCard::path)
            .def_prop_ro("project_uuid", [](const project::ProjectInspectorCard& value) {
                return value.project_uuid.to_string();
            })
            .def_prop_ro("file_uuid", [](const project::ProjectInspectorCard& value) {
                return value.file_uuid.to_string();
            })
            .def_prop_ro("commit_uuid", [](const project::ProjectInspectorCard& value) {
                return value.commit_uuid.to_string();
            })
            .def_ro("generation", &project::ProjectInspectorCard::generation)
            .def_ro("created_at_unix_ns", &project::ProjectInspectorCard::created_at_unix_ns)
            .def_ro("saved_at_unix_ns", &project::ProjectInspectorCard::saved_at_unix_ns)
            .def_ro("physical_file_size", &project::ProjectInspectorCard::physical_file_size)
            .def_ro("role", &project::ProjectInspectorCard::role)
            .def_ro("open_state", &project::ProjectInspectorCard::open_state)
            .def_ro("validation_scope", &project::ProjectInspectorCard::validation_scope)
            .def_ro("has_preview", &project::ProjectInspectorCard::has_preview)
            .def_ro("preview_bytes", &project::ProjectInspectorCard::preview_bytes)
            .def_ro("preview_width", &project::ProjectInspectorCard::preview_width)
            .def_ro("preview_height", &project::ProjectInspectorCard::preview_height)
            .def_ro("title", &project::ProjectInspectorCard::title)
            .def_ro("min_reader_version", &project::ProjectInspectorCard::min_reader_version)
            .def_ro("min_safe_writer_version", &project::ProjectInspectorCard::min_safe_writer_version)
            .def_ro("commit_kind", &project::ProjectInspectorCard::commit_kind)
            .def_ro("diagnostic", &project::ProjectInspectorCard::diagnostic);

        nb::class_<project::ProjectThumbnailSourceAvailability>(
            m, "ProjectThumbnailSourceAvailability")
            .def_ro("first_dataset_image",
                    &project::ProjectThumbnailSourceAvailability::first_dataset_image)
            .def_ro("first_embedded_image",
                    &project::ProjectThumbnailSourceAvailability::first_embedded_image);

        nb::class_<project::ProjectInspectorSave>(m, "ProjectInspectorSave")
            .def_ro("sequence", &project::ProjectInspectorSave::sequence)
            .def_ro("generation", &project::ProjectInspectorSave::generation)
            .def_ro("kind", &project::ProjectInspectorSave::kind)
            .def_ro("saved_at_unix_ns", &project::ProjectInspectorSave::saved_at_unix_ns)
            .def_ro("bytes_added", &project::ProjectInspectorSave::bytes_added)
            .def_ro("holds_checkpoint", &project::ProjectInspectorSave::holds_checkpoint)
            .def_ro("checkpoint_iteration", &project::ProjectInspectorSave::checkpoint_iteration)
            .def_ro("planned_iterations", &project::ProjectInspectorSave::planned_iterations)
            .def_ro("strategy", &project::ProjectInspectorSave::strategy)
            .def_ro("gaussians", &project::ProjectInspectorSave::gaussians)
            .def_ro("operation", &project::ProjectInspectorSave::operation)
            .def_ro("source_save_generation", &project::ProjectInspectorSave::source_save_generation)
            .def_ro("source_saved_at_unix_ns", &project::ProjectInspectorSave::source_saved_at_unix_ns)
            .def_ro("source_save_kind", &project::ProjectInspectorSave::source_save_kind);

        nb::class_<project::ProjectInspectorChapter>(m, "ProjectInspectorChapter")
            .def_prop_ro("fourcc", [](const project::ProjectInspectorChapter& value) {
                return value.fourcc.to_string();
            })
            .def_prop_ro("instance_uuid", [](const project::ProjectInspectorChapter& value) {
                return value.instance_uuid.to_string();
            })
            .def_ro("row_kind", &project::ProjectInspectorChapter::row_kind)
            .def_ro("compression", &project::ProjectInspectorChapter::compression)
            .def_ro("stored_bytes", &project::ProjectInspectorChapter::stored_bytes)
            .def_ro("uncompressed_bytes", &project::ProjectInspectorChapter::uncompressed_bytes)
            .def_ro("source_generation", &project::ProjectInspectorChapter::source_generation);

        nb::class_<project::ProjectInspectorCheckpoint>(m, "ProjectInspectorCheckpoint")
            .def_prop_ro("instance_uuid", [](const project::ProjectInspectorCheckpoint& value) {
                return value.instance_uuid.to_string();
            })
            .def_ro("source_generation", &project::ProjectInspectorCheckpoint::source_generation)
            .def_ro("iteration", &project::ProjectInspectorCheckpoint::iteration)
            .def_ro("gaussians", &project::ProjectInspectorCheckpoint::gaussians)
            .def_ro("sh_degree", &project::ProjectInspectorCheckpoint::sh_degree)
            .def_ro("binds_scene_graph", &project::ProjectInspectorCheckpoint::binds_scene_graph)
            .def_ro("header_reachable", &project::ProjectInspectorCheckpoint::header_reachable)
            .def_ro("retained", &project::ProjectInspectorCheckpoint::retained);

        nb::class_<project::ProjectInspectorSceneGraph>(m, "ProjectInspectorSceneGraph")
            .def_ro("node_counts_by_type", &project::ProjectInspectorSceneGraph::node_counts_by_type)
            .def_ro("dataset_node_name", &project::ProjectInspectorSceneGraph::dataset_node_name)
            .def_prop_ro("training_node_id", [](const project::ProjectInspectorSceneGraph& value) -> std::optional<std::string> {
                if (!value.training_node_id) {
                    return std::nullopt;
                }
                return value.training_node_id->to_string();
            });

        nb::class_<project::ProjectInspectorParameters>(m, "ProjectInspectorParameters")
            .def_ro("active_strategy", &project::ProjectInspectorParameters::active_strategy)
            .def_ro("planned_iterations", &project::ProjectInspectorParameters::planned_iterations)
            .def_ro("embedded_dataset_present", &project::ProjectInspectorParameters::embedded_dataset_present)
            .def_ro("embedded_dataset_complete", &project::ProjectInspectorParameters::embedded_dataset_complete)
            .def_ro("embedded_images", &project::ProjectInspectorParameters::embedded_images)
            .def_ro("embedded_normals", &project::ProjectInspectorParameters::embedded_normals)
            .def_ro("embedded_sparse", &project::ProjectInspectorParameters::embedded_sparse);

        nb::class_<project::ProjectInspectorReference>(m, "ProjectInspectorReference")
            .def_ro("key", &project::ProjectInspectorReference::key)
            .def_ro("kind", &project::ProjectInspectorReference::kind)
            .def_ro("path", &project::ProjectInspectorReference::path)
            .def_ro("reachable", &project::ProjectInspectorReference::reachable);

        nb::class_<project::MetricHistorySample>(m, "ProjectMetricHistorySample")
            .def_ro("iteration", &project::MetricHistorySample::iteration)
            .def_ro("value", &project::MetricHistorySample::value);
        nb::class_<project::LastEvaluationMetrics>(m, "ProjectLastEvaluationMetrics")
            .def_ro("iteration", &project::LastEvaluationMetrics::iteration)
            .def_ro("psnr", &project::LastEvaluationMetrics::psnr)
            .def_ro("ssim", &project::LastEvaluationMetrics::ssim);
        nb::class_<project::ProjectInspectorMetrics>(m, "ProjectInspectorMetrics")
            .def_ro("loss_samples", &project::ProjectInspectorMetrics::loss_samples)
            .def_ro("psnr_samples", &project::ProjectInspectorMetrics::psnr_samples)
            .def_ro("last_loss", &project::ProjectInspectorMetrics::last_loss)
            .def_ro("last_psnr", &project::ProjectInspectorMetrics::last_psnr)
            .def_ro("last_evaluation", &project::ProjectInspectorMetrics::last_evaluation);

        nb::class_<project::ProjectInspectorDetails>(m, "ProjectInspectorDetails")
            .def_ro("card", &project::ProjectInspectorDetails::card)
            .def_ro("storage", &project::ProjectInspectorDetails::storage)
            .def_ro("save_history", &project::ProjectInspectorDetails::save_history)
            .def_ro("chapters", &project::ProjectInspectorDetails::chapters)
            .def_ro("manifest", &project::ProjectInspectorDetails::manifest)
            .def_ro("license", &project::ProjectInspectorDetails::license)
            .def_ro("scene_graph", &project::ProjectInspectorDetails::scene_graph)
            .def_ro("parameters", &project::ProjectInspectorDetails::parameters)
            .def_ro("retained_checkpoints", &project::ProjectInspectorDetails::retained_checkpoints)
            .def_ro("references", &project::ProjectInspectorDetails::references)
            .def_ro("metrics", &project::ProjectInspectorDetails::metrics)
            .def_ro("autosave_sidecar_present", &project::ProjectInspectorDetails::autosave_sidecar_present)
            .def_ro("chapters_requiring_full_read", &project::ProjectInspectorDetails::chapters_requiring_full_read);

        nb::enum_<project::ProjectVerificationStatus>(m, "ProjectVerificationStatus")
            .value("VERIFIED", project::ProjectVerificationStatus::Verified)
            .value("CANCELED", project::ProjectVerificationStatus::Canceled)
            .value("FAILED", project::ProjectVerificationStatus::Failed);

        nb::class_<project::ProjectVerificationResult>(m, "ProjectVerificationResult")
            .def_ro("status", &project::ProjectVerificationResult::status)
            .def_ro("verified_chunks", &project::ProjectVerificationResult::verified_chunks)
            .def_ro("first_mismatch", &project::ProjectVerificationResult::first_mismatch);

        nb::class_<project::ProjectReduceCheckpoint>(m, "ProjectReduceCheckpoint")
            .def_prop_ro("instance_uuid", [](const project::ProjectReduceCheckpoint& value) {
                return value.instance_uuid.to_string();
            })
            .def_ro("iteration", &project::ProjectReduceCheckpoint::iteration)
            .def_ro("bytes", &project::ProjectReduceCheckpoint::bytes)
            .def_ro("scng_bound", &project::ProjectReduceCheckpoint::scng_bound);

        nb::class_<project::ProjectReduceDatasetPayload>(m, "ProjectReduceDatasetPayload")
            .def_prop_ro("chunk_uuid", [](const project::ProjectReduceDatasetPayload& value) {
                return value.chunk_uuid.to_string();
            })
            .def_ro("rel_path", &project::ProjectReduceDatasetPayload::rel_path)
            .def_ro("kind", &project::ProjectReduceDatasetPayload::kind)
            .def_ro("bytes", &project::ProjectReduceDatasetPayload::bytes)
            .def_ro("external_replacement_validated", &project::ProjectReduceDatasetPayload::external_replacement_validated);

        nb::class_<project::ProjectReduceProjection>(m, "ProjectReduceProjection")
            .def_ro("enabled", &project::ProjectReduceProjection::enabled)
            .def_ro("allowed", &project::ProjectReduceProjection::allowed)
            .def_ro("reclaimable_bytes", &project::ProjectReduceProjection::reclaimable_bytes)
            .def_ro("projected_size", &project::ProjectReduceProjection::projected_size);

        nb::class_<project::ProjectReducePlan>(m, "ProjectReducePlan")
            .def_ro("path", &project::ProjectReducePlan::path)
            .def_prop_ro("input_commit_uuid", [](const project::ProjectReducePlan& value) {
                return value.input_commit_uuid.to_string();
            })
            .def_ro("physical_size", &project::ProjectReducePlan::physical_size)
            .def_ro("tombstone_bytes", &project::ProjectReducePlan::tombstone_bytes)
            .def_ro("superseded_rows", &project::ProjectReducePlan::superseded_rows)
            .def_ro("retained_checkpoints", &project::ProjectReducePlan::retained_checkpoints)
            .def_ro("embedded_dataset", &project::ProjectReducePlan::embedded_dataset)
            .def_ro("drop_checkpoints", &project::ProjectReducePlan::drop_checkpoints)
            .def_ro("drop_embedded_dataset", &project::ProjectReducePlan::drop_embedded_dataset)
            .def_ro("compact", &project::ProjectReducePlan::compact);

        nb::class_<project::ProjectReduceResult>(m, "ProjectReduceResult")
            .def_ro("card", &project::ProjectReduceResult::card)
            .def_ro("checkpoints_removed", &project::ProjectReduceResult::checkpoints_removed)
            .def_ro("dataset_images_removed", &project::ProjectReduceResult::dataset_images_removed)
            .def_ro("dataset_normals_removed", &project::ProjectReduceResult::dataset_normals_removed)
            .def_ro("dataset_sparse_removed", &project::ProjectReduceResult::dataset_sparse_removed)
            .def_ro("bytes_reclaimed", &project::ProjectReduceResult::bytes_reclaimed)
            .def_ro("recovery_copy", &project::ProjectReduceResult::recovery_copy);

        nb::class_<project::DatasetEmbedResult>(m, "DatasetEmbedResult")
            .def_ro("card", &project::DatasetEmbedResult::card)
            .def_ro("images_embedded", &project::DatasetEmbedResult::images_embedded)
            .def_ro("normals_embedded", &project::DatasetEmbedResult::normals_embedded)
            .def_ro("sparse_embedded", &project::DatasetEmbedResult::sparse_embedded)
            .def_ro("bytes_embedded", &project::DatasetEmbedResult::bytes_embedded);

        nb::class_<project::DatasetReferenceResult>(m, "DatasetReferenceResult")
            .def_ro("card", &project::DatasetReferenceResult::card)
            .def_prop_ro("reference_uuid", [](const project::DatasetReferenceResult& value) {
                return value.reference_uuid.to_string();
            })
            .def_ro("content_replaced", &project::DatasetReferenceResult::content_replaced);

        nb::enum_<project::ProjectExportFormat>(m, "ProjectExportFormat")
            .value("PLY", project::ProjectExportFormat::Ply)
            .value("SOG", project::ProjectExportFormat::Sog)
            .value("SSOG", project::ProjectExportFormat::Ssog)
            .value("SPZ", project::ProjectExportFormat::Spz);

        nb::class_<project::ProjectExportResult>(m, "ProjectExportResult")
            .def_ro("destination", &project::ProjectExportResult::destination)
            .def_ro("format", &project::ProjectExportResult::format)
            .def_ro("gaussian_count", &project::ProjectExportResult::gaussian_count)
            .def_ro("bytes_written", &project::ProjectExportResult::bytes_written);

        nb::class_<project::ProjectRepairResult>(m, "ProjectRepairResult")
            .def_ro("card", &project::ProjectRepairResult::card)
            .def_ro("saves_recovered", &project::ProjectRepairResult::saves_recovered);

        m.def("classify_project", [](const std::filesystem::path& path) {
            std::optional<project::OpenClassification> result;
            {
                nb::gil_scoped_release release;
                project::ReaderOptions options;
                options.allow_unsupported_inspection = true;
                result = project::ProjectReader::classify(path, options);
            }
            return *result; }, nb::arg("path"), "Classify a .licht path without throwing for damaged heads.");

        m.def("project_storage_stats", [](const std::filesystem::path& path) {
            std::optional<lfs::Result<project::ProjectStorageStats>> result;
            {
                nb::gil_scoped_release release;
                result = project::project_storage_stats(path);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"));

        m.def("run_project_operation", [](const std::filesystem::path& path, const std::string& project_uuid, const std::string& commit_uuid, nb::callable operation) {
            const auto expected_project = parse_reference_uuid(project_uuid);
            const auto expected_commit = commit_uuid.empty() ? lfs::core::Uuid{} : parse_reference_uuid(commit_uuid);
            unwrap(project::run_project_operation(path, expected_project, expected_commit, [&operation] { operation(); })); }, nb::arg("path"), nb::arg("project_uuid"), nb::arg("commit_uuid"), nb::arg("operation"));

        m.def("backup_project_file", [](const std::filesystem::path& path) {
            nb::gil_scoped_release release;
            return unwrap(project::backup_project_file(path)); }, nb::arg("path"));

        m.def("restore_project_backup", [](const std::filesystem::path& path, const std::filesystem::path& backup, const std::string& project_uuid, const std::string& commit_uuid) {
            const auto project = parse_reference_uuid(project_uuid);
            const auto commit = parse_reference_uuid(commit_uuid);
            nb::gil_scoped_release release;
            unwrap(project::restore_project_backup(path, backup, project, commit)); }, nb::arg("path"), nb::arg("backup"), nb::arg("project_uuid"), nb::arg("commit_uuid"));

        m.def("inspect_project_card", [](const std::filesystem::path& path) {
            std::optional<lfs::Result<project::ProjectInspectorCard>> result;
            {
                nb::gil_scoped_release release;
                result = project::inspect_project_card(path);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"));

        m.def("project_content_stamp", [](const std::filesystem::path& path) {
            nb::gil_scoped_release release;
            return project::project_content_stamp(path); }, nb::arg("path"));

        m.def("inspect_project_details", [](const std::filesystem::path& path, const std::uint64_t checkpoint_byte_budget) {
            std::optional<lfs::Result<project::ProjectInspectorDetails>> result;
            {
                nb::gil_scoped_release release;
                result = project::inspect_project_details(path, checkpoint_byte_budget);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"), nb::arg("checkpoint_byte_budget") = 8ull * 1024 * 1024);

        m.def("read_preview", [](const std::filesystem::path& path) {
            std::optional<lfs::Result<std::vector<std::byte>>> result;
            {
                nb::gil_scoped_release release;
                result = project::read_project_preview(path);
            }
            const auto bytes = unwrap(std::move(*result));
            return nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size()); }, nb::arg("path"));

        m.def("restore_save", [](const std::filesystem::path& path, const std::uint64_t generation, const std::filesystem::path& destination) {
            std::optional<lfs::Result<project::ProjectInspectorCard>> result;
            {
                nb::gil_scoped_release release;
                result = project::restore_save(path, generation, destination);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"), nb::arg("generation"), nb::arg("destination"));

        m.def("rebind_checkpoint", [](const std::filesystem::path& path, const std::string& checkpoint_instance_uuid) {
            const auto uuid = parse_reference_uuid(checkpoint_instance_uuid);
            std::optional<lfs::Result<project::ProjectInspectorCard>> result;
            {
                nb::gil_scoped_release release;
                result = project::rebind_checkpoint(path, uuid);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"), nb::arg("checkpoint_instance_uuid"));

        m.def("compact_project_file", [](const std::filesystem::path& path, nb::object progress, nb::object cancel) {
            PyProgressCallback progress_callback{std::move(progress)};
            PyCancelCallback cancel_callback{std::move(cancel)};
            std::optional<lfs::Result<project::ProjectInspectorCard>> result;
            {
                nb::gil_scoped_release release;
                result = project::compact_project_file(
                    path,
                    progress_callback.callback && !progress_callback.callback.is_none()
                        ? project::ProjectOperationProgress(progress_callback)
                        : project::ProjectOperationProgress{},
                    cancel_callback.callback && !cancel_callback.callback.is_none()
                        ? project::ProjectOperationCancel(cancel_callback)
                        : project::ProjectOperationCancel{});
            }
            return unwrap(std::move(*result)); }, nb::arg("path"), nb::arg("progress") = nb::none(), nb::arg("cancel") = nb::none());

        m.def("clean_project_file", [](const std::filesystem::path& path, const std::filesystem::path& destination, const std::string& expected_commit, nb::object progress, nb::object cancel) {
            const auto expected = expected_commit.empty() ? lfs::core::Uuid{} : parse_reference_uuid(expected_commit);
            PyProgressCallback progress_callback{std::move(progress)};
            PyCancelCallback cancel_callback{std::move(cancel)};
            nb::gil_scoped_release release;
            return unwrap(project::clean_project_file(path, destination, expected,
                progress_callback.callback && !progress_callback.callback.is_none() ? project::ProjectOperationProgress(progress_callback) : project::ProjectOperationProgress{},
                cancel_callback.callback && !cancel_callback.callback.is_none() ? project::ProjectOperationCancel(cancel_callback) : project::ProjectOperationCancel{})); }, nb::arg("path"), nb::arg("destination") = "", nb::arg("expected_commit") = "", nb::arg("progress") = nb::none(), nb::arg("cancel") = nb::none());

        m.def("plan_reduce_size", [](const std::filesystem::path& path) {
            std::optional<lfs::Result<project::ProjectReducePlan>> result;
            {
                nb::gil_scoped_release release;
                result = project::plan_reduce_size(path);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"));

        m.def("undo_contents_removal", [](const std::filesystem::path& path, const std::string& id) {
            std::optional<lfs::Result<project::ProjectInspectorCard>> result;
            {
                nb::gil_scoped_release release;
                result = project::undo_contents_removal(path, id);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"), nb::arg("id"));

        m.def("reduce_size", [](const std::filesystem::path& path, const nb::dict& options, nb::object progress, nb::object cancel) {
            const auto read_option = [&](const char* name, const bool fallback) {
                return options.contains(name) ? nb::cast<bool>(options[name]) : fallback;
            };
            const bool drop_unbound_checkpoints = read_option("drop_unbound_checkpoints", true);
            const bool drop_embedded_dataset = read_option("drop_embedded_dataset", false);
            project::ProjectReduceSelection selection;
            selection.compact = read_option("compact", true);
            selection.thumbnail = read_option("drop_thumbnail", false);
            selection.metrics = read_option("drop_metrics", false);
            if (options.contains("save_generation")) selection.save_generation = nb::cast<std::uint64_t>(options["save_generation"]);
            if (options.contains("checkpoint_uuid")) selection.checkpoint = parse_reference_uuid(nb::cast<std::string>(options["checkpoint_uuid"]));
            PyProgressCallback progress_callback{std::move(progress)};
            PyCancelCallback cancel_callback{std::move(cancel)};
            std::optional<lfs::Result<project::ProjectReduceResult>> result;
            {
                nb::gil_scoped_release release;
                result = project::reduce_size(
                    path, drop_unbound_checkpoints, drop_embedded_dataset,
                    progress_callback.callback && !progress_callback.callback.is_none()
                        ? project::ProjectOperationProgress(progress_callback)
                        : project::ProjectOperationProgress{},
                    cancel_callback.callback && !cancel_callback.callback.is_none()
                        ? project::ProjectOperationCancel(cancel_callback)
                        : project::ProjectOperationCancel{}, selection);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"), nb::arg("options") = nb::dict(), nb::arg("progress") = nb::none(), nb::arg("cancel") = nb::none());

        m.def("embed_dataset_file", [](const std::filesystem::path& path, nb::object progress, nb::object cancel) {
            PyProgressCallback progress_callback{std::move(progress)};
            PyCancelCallback cancel_callback{std::move(cancel)};
            std::optional<lfs::Result<project::DatasetEmbedResult>> result;
            {
                nb::gil_scoped_release release;
                result = project::embed_dataset_file(
                    path,
                    progress_callback.callback && !progress_callback.callback.is_none()
                        ? project::ProjectOperationProgress(progress_callback)
                        : project::ProjectOperationProgress{},
                    cancel_callback.callback && !cancel_callback.callback.is_none()
                        ? project::ProjectOperationCancel(cancel_callback)
                        : project::ProjectOperationCancel{});
            }
            return unwrap(std::move(*result)); }, nb::arg("path"), nb::arg("progress") = nb::none(), nb::arg("cancel") = nb::none());

        m.def("set_dataset_reference", [](const std::filesystem::path& path, const std::filesystem::path& dataset_dir, const bool accept_content_change) {
            std::optional<lfs::Result<project::DatasetReferenceResult>> result;
            {
                nb::gil_scoped_release release;
                result = project::set_dataset_reference(path, dataset_dir, accept_content_change);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"), nb::arg("dataset_dir"), nb::arg("accept_content_change") = false);

        m.def("export_project_as", [](const std::filesystem::path& path, const std::string& format, const std::filesystem::path& destination, nb::object progress, nb::object cancel) {
            project::ProjectExportFormat export_format;
            if (format == "ply" || format == "PLY") export_format = project::ProjectExportFormat::Ply;
            else if (format == "sog" || format == "SOG") export_format = project::ProjectExportFormat::Sog;
            else if (format == "ssog" || format == "SSOG") export_format = project::ProjectExportFormat::Ssog;
            else if (format == "spz" || format == "SPZ") export_format = project::ProjectExportFormat::Spz;
            else throw nb::value_error("format must be PLY, SOG, SSOG, or SPZ");
            PyProgressCallback progress_callback{std::move(progress)};
            PyCancelCallback cancel_callback{std::move(cancel)};
            std::optional<lfs::Result<project::ProjectExportResult>> result;
            {
                nb::gil_scoped_release release;
                result = project::export_project_as(
                    path, export_format, destination,
                    progress_callback.callback && !progress_callback.callback.is_none()
                        ? project::ProjectOperationProgress(progress_callback)
                        : project::ProjectOperationProgress{},
                    cancel_callback.callback && !cancel_callback.callback.is_none()
                        ? project::ProjectOperationCancel(cancel_callback)
                        : project::ProjectOperationCancel{});
            }
            return unwrap(std::move(*result)); }, nb::arg("path"), nb::arg("format"), nb::arg("destination"), nb::arg("progress") = nb::none(), nb::arg("cancel") = nb::none());

        m.def("repair_project", [](const std::filesystem::path& path, const std::filesystem::path& destination, const std::string& expected_project) {
            const auto expected = expected_project.empty() ? lfs::core::Uuid{} : parse_reference_uuid(expected_project);
            std::optional<lfs::Result<project::ProjectRepairResult>> result;
            {
                nb::gil_scoped_release release;
                result = project::repair_project(path, destination, expected);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"), nb::arg("destination"), nb::arg("expected_project") = "");

        m.def("verify_project_file", [](const std::filesystem::path& path, nb::object progress, nb::object cancel) {
            PyProgressCallback progress_callback{std::move(progress)};
            PyCancelCallback cancel_callback{std::move(cancel)};
            std::optional<lfs::Result<project::ProjectVerificationResult>> result;
            {
                nb::gil_scoped_release release;
                result = project::verify_project_file(
                    path,
                    progress_callback.callback && !progress_callback.callback.is_none()
                        ? project::ProjectOperationProgress(progress_callback)
                        : project::ProjectOperationProgress{},
                    cancel_callback.callback && !cancel_callback.callback.is_none()
                        ? project::ProjectOperationCancel(cancel_callback)
                        : project::ProjectOperationCancel{});
            }
            return unwrap(std::move(*result)); }, nb::arg("path"), nb::arg("progress") = nb::none(), nb::arg("cancel") = nb::none());

        m.def("set_project_preview", [](const std::filesystem::path& path, const nb::bytes& png) {
            std::optional<lfs::Result<project::ProjectInspectorCard>> result;
            {
                nb::gil_scoped_release release;
                result = project::set_project_preview(
                    path,
                    std::span<const std::byte>(
                        static_cast<const std::byte*>(png.data()), png.size()));
            }
            return unwrap(std::move(*result)); }, nb::arg("path"), nb::arg("png_bytes"));

        m.def("encode_preview_from_image_file", [](const std::filesystem::path& image_path) {
            std::optional<lfs::Result<std::vector<std::byte>>> result;
            {
                nb::gil_scoped_release release;
                result = project::encode_preview_from_image_file(image_path);
            }
            const auto bytes = unwrap(std::move(*result));
            return nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size()); }, nb::arg("image_path"));

        m.def("encode_preview_from_first_dataset_image", [](const std::filesystem::path& path) {
            std::optional<lfs::Result<std::vector<std::byte>>> result;
            {
                nb::gil_scoped_release release;
                result = project::encode_preview_from_first_dataset_image(path);
            }
            const auto bytes = unwrap(std::move(*result));
            return nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size()); }, nb::arg("path"));

        m.def("encode_preview_from_first_embedded_image", [](const std::filesystem::path& path) {
            std::optional<lfs::Result<std::vector<std::byte>>> result;
            {
                nb::gil_scoped_release release;
                result = project::encode_preview_from_first_embedded_image(path);
            }
            const auto bytes = unwrap(std::move(*result));
            return nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size()); }, nb::arg("path"));

        m.def("inspect_project_thumbnail_sources", [](const std::filesystem::path& path) {
            std::optional<lfs::Result<project::ProjectThumbnailSourceAvailability>> result;
            {
                nb::gil_scoped_release release;
                result = project::inspect_project_thumbnail_sources(path);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"));

        m.def("preview_from_first_dataset_image", [](const std::filesystem::path& path) {
            std::optional<lfs::Result<project::ProjectInspectorCard>> result;
            {
                nb::gil_scoped_release release;
                result = project::preview_from_first_dataset_image(path);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"));

        m.def("preview_from_first_embedded_image", [](const std::filesystem::path& path) {
            std::optional<lfs::Result<project::ProjectInspectorCard>> result;
            {
                nb::gil_scoped_release release;
                result = project::preview_from_first_embedded_image(path);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"));

        m.def("preview_from_image_file", [](const std::filesystem::path& project_path, const std::filesystem::path& image_path) {
            std::optional<lfs::Result<project::ProjectInspectorCard>> result;
            {
                nb::gil_scoped_release release;
                result = project::preview_from_image_file(project_path, image_path);
            }
            return unwrap(std::move(*result)); }, nb::arg("project_path"), nb::arg("image_path"));

        m.def("set_project_license", [](const std::filesystem::path& path, const std::string& identifier, const std::string& notice) {
            std::optional<lfs::Result<project::ProjectInspectorCard>> result;
            {
                nb::gil_scoped_release release;
                result = project::set_project_license(path, identifier, notice);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"), nb::arg("identifier"), nb::arg("notice") = "");

        m.def("clear_project_license", [](const std::filesystem::path& path) {
            std::optional<lfs::Result<project::ProjectInspectorCard>> result;
            {
                nb::gil_scoped_release release;
                result = project::clear_project_license(path);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"));

        m.def("set_project_title", [](const std::filesystem::path& path, const std::string& title) {
            std::optional<lfs::Result<project::ProjectInspectorCard>> result;
            {
                nb::gil_scoped_release release;
                result = project::set_project_title(path, title);
            }
            return unwrap(std::move(*result)); }, nb::arg("path"), nb::arg("title"));

        m.def(
            "inspect_project",
            [](const std::filesystem::path& path, const bool resolve_preview_fallback) {
                project::ReaderOptions options;
                options.allow_unsupported_inspection = true;
                std::optional<lfs::Result<project::ProjectReader>> opened;
                std::string fallback_preview_path;
                std::uint32_t preview_width = 0;
                std::uint32_t preview_height = 0;
                project::ProjectFilterFacts facts;
                {
                    nb::gil_scoped_release release;
                    opened = project::ProjectReader::open(path, options);
                    if (opened && opened->has_value())
                        facts = project::inspect_project_filter_facts(**opened);
                    if (resolve_preview_fallback && opened && opened->has_value() &&
                        !(**opened).preview().has_value()) {
                        const auto& reader = **opened;
                        const auto project_uuid =
                            reader.superblock().project_uuid;
                        const auto* proj = reader.find(
                            project::FOURCC_PROJ, project_uuid);
                        const auto* refs = reader.find(
                            project::FOURCC_REFS, project_uuid);
                        const auto* prms = reader.find(
                            project::FOURCC_PRMS, project_uuid);
                        if (proj && refs && prms) {
                            auto proj_bytes = reader.read_chunk(*proj);
                            auto refs_bytes = reader.read_chunk(*refs);
                            auto prms_bytes = reader.read_chunk(*prms);
                            if (proj_bytes && refs_bytes && prms_bytes) {
                                auto project_chapter =
                                    project::ProjectChapter::from_bytes(*proj_bytes);
                                auto references_chapter =
                                    project::ReferencesChapter::from_bytes(*refs_bytes);
                                auto parameters_chapter =
                                    project::ParametersChapter::from_bytes(*prms_bytes);
                                if (project_chapter && references_chapter &&
                                    parameters_chapter) {
                                    if (const auto first =
                                            project::first_dataset_image(
                                                *project_chapter, *references_chapter,
                                                *parameters_chapter,
                                                reader.path().parent_path())) {
                                        fallback_preview_path =
                                            lfs::core::path_to_utf8(*first);
                                    }
                                }
                            }
                        }
                    }
                    if (opened && opened->has_value()) {
                        std::tie(preview_width, preview_height) =
                            project::project_preview_dimensions(**opened);
                        if ((preview_width == 0 || preview_height == 0) &&
                            !fallback_preview_path.empty()) {
                            try {
                                const auto info = lfs::core::get_image_info(
                                    lfs::core::utf8_to_path(fallback_preview_path));
                                const int width = std::get<0>(info);
                                const int height = std::get<1>(info);
                                preview_width = width > 0
                                                    ? static_cast<std::uint32_t>(width)
                                                    : 0;
                                preview_height = height > 0
                                                     ? static_cast<std::uint32_t>(height)
                                                     : 0;
                            } catch (...) {
                                // LFS-CENSUS-OK(empty-catch): optional preview metadata must not make project inspection fail.
                                preview_width = 0;
                                preview_height = 0;
                            }
                        }
                    }
                }
                auto reader = unwrap(std::move(*opened));
                return PyProjectInspection{
                    .project_uuid = reader.superblock().project_uuid.to_string(),
                    .file_uuid = reader.superblock().file_uuid.to_string(),
                    .commit_uuid = reader.commit().commit_uuid.to_string(),
                    .generation = reader.commit().generation,
                    .created_at_unix_ns = reader.superblock().creation_time_unix_ns,
                    .saved_at_unix_ns = reader.commit().wallclock_unix_ns,
                    .physical_file_size = reader.physical_file_size(),
                    .role = reader.superblock().role,
                    .open_state = reader.open_state(),
                    .has_preview = reader.preview().has_value(),
                    .has_checkpoint = facts.has_checkpoint,
                    .has_dataset = facts.has_dataset,
                    .preview_width = preview_width,
                    .preview_height = preview_height,
                    .fallback_preview_path = std::move(fallback_preview_path),
                };
            },
            nb::arg("path"),
            nb::arg("resolve_preview_fallback") = true,
            "Inspect validated .licht metadata and lightweight project contents.");

        nb::class_<PyLoadResult>(m, "LoadResult")
            .def_prop_ro("splat_data", &PyLoadResult::get_splat_data, "Loaded splat data, or None")
            .def_prop_ro(
                "scene_center", [](const PyLoadResult& r) { return r.scene_center; }, "Scene center [3] tensor")
            .def_prop_ro(
                "loader_used", [](const PyLoadResult& r) { return r.loader_used; }, "Name of loader that was used")
            .def_prop_ro(
                "load_time_ms", [](const PyLoadResult& r) { return r.load_time_ms; }, "Load time in milliseconds")
            .def_prop_ro(
                "warnings", [](const PyLoadResult& r) { return r.warnings; }, "List of warning messages from loading")
            .def_prop_ro("cameras", &PyLoadResult::get_cameras, "Camera dataset, or None")
            .def_prop_ro("point_cloud", &PyLoadResult::get_point_cloud, "Point cloud, or None")
            .def_prop_ro("is_dataset", &PyLoadResult::is_dataset, "Whether loaded data is a dataset with cameras");

        m.def(
            "load",
            [](const std::filesystem::path& path, std::optional<std::string> format,
               std::optional<int> resize_factor, std::optional<int> max_width,
               std::optional<std::string> images_folder, nb::object progress,
               std::optional<int> min_track_length) -> PyLoadResult {
                auto loader = io::Loader::create();

                io::LoadOptions options;
                if (resize_factor)
                    options.resize_factor = *resize_factor;
                if (max_width)
                    options.max_width = *max_width;
                if (images_folder)
                    options.images_folder = *images_folder;
                if (min_track_length)
                    options.min_track_length = *min_track_length;

                if (progress && !progress.is_none()) {
                    PyProgressCallback py_progress{nb::cast<nb::object>(progress)};
                    options.progress = [py_progress](float p, const std::string& msg) {
                        py_progress(p, msg);
                    };
                }

                io::Result<io::LoadResult> result;
                {
                    nb::gil_scoped_release release;
                    result = loader->load(path, options);
                }
                if (!result) {
                    throw_io_error(result.error(),
                                   std::format("Failed to load '{}'", lfs::core::path_to_utf8(path)));
                }

                PyLoadResult py_result;
                py_result.loader_used = result->loader_used;
                py_result.load_time_ms = result->load_time.count();
                py_result.warnings = result->warnings;
                py_result.scene_center = PyTensor(result->scene_center, false);

                if (std::holds_alternative<std::shared_ptr<core::SplatData>>(result->data)) {
                    py_result.splat_data = std::get<std::shared_ptr<core::SplatData>>(result->data);
                } else {
                    auto& scene = std::get<io::LoadedScene>(result->data);
                    py_result.cameras = std::move(scene.cameras);
                    py_result.point_cloud = std::move(scene.point_cloud);
                }

                return py_result;
            },
            nb::arg("path"), nb::arg("format") = nb::none(), nb::arg("resize_factor") = nb::none(),
            nb::arg("max_width") = nb::none(), nb::arg("images_folder") = nb::none(),
            nb::arg("progress") = nb::none(),
            nb::arg("min_track_length") = nb::none(),
            "Load a scene or splat file from path");

        m.def(
            "load_point_cloud",
            [](const std::filesystem::path& path) -> nb::tuple {
                std::expected<lfs::core::PointCloud, std::string> result;
                {
                    nb::gil_scoped_release release;
                    result = io::load_ply_point_cloud(path);
                }
                if (!result)
                    throw lfs::Exception(lfs::make_error({
                        .code = lfs::ErrorCode::Internal,
                        .domain = lfs::ErrorDomain::IO,
                        .severity = lfs::Severity::Error,
                        .user_message = std::format("Failed to load point cloud: {}", result.error()),
                        .detection = LFS_SOURCE_SITE_CURRENT(),
                    }));
                return nb::make_tuple(PyTensor(result->means, true), PyTensor(result->colors, true));
            },
            nb::arg("path"),
            "Load a PLY as point cloud, returns (means [N,3], colors [N,3]) tensors");

        m.def(
            "save_ply",
            [](const PySplatData& data, const std::filesystem::path& path, bool binary,
               nb::object progress, nb::object extra_attributes, bool include_provenance) {
                io::PlySaveOptions options;
                options.output_path = path;
                options.binary = binary;
                options.extra_attributes = parse_extra_ply_attributes(extra_attributes, path);
                options.provenance = include_provenance ? core::make_provenance_stamp()
                                                        : core::make_minimal_provenance_stamp();

                if (progress && !progress.is_none()) {
                    PyExportProgressCallback py_progress{nb::cast<nb::object>(progress)};
                    options.progress_callback = [py_progress](float p, const std::string& stage) -> bool {
                        return py_progress(p, stage);
                    };
                }

                auto result = [&] {
                    nb::gil_scoped_release release;
                    return io::save_ply(*data.data(), options);
                }();
                if (!result)
                    throw_io_error(result.error(), "Failed to save PLY");
            },
            nb::arg("data"), nb::arg("path"), nb::arg("binary") = true, nb::arg("progress") = nb::none(),
            nb::arg("extra_attributes") = nb::none(),
            nb::arg("include_provenance") = true,
            "Save splat data as PLY file with optional extra per-vertex float attributes. "
            "include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.");

        m.def(
            "save_point_cloud_ply",
            [](const PyPointCloud& pc, const std::filesystem::path& path, nb::object extra_attributes,
               bool include_provenance) {
                if (!pc.data())
                    throw_invalid_io_argument("Point cloud data must not be null");
                io::PlySaveOptions options;
                options.output_path = path;
                options.binary = true;
                options.extra_attributes = parse_extra_ply_attributes(extra_attributes, path);
                options.provenance = include_provenance ? core::make_provenance_stamp()
                                                        : core::make_minimal_provenance_stamp();
                auto result = [&] {
                    nb::gil_scoped_release release;
                    return io::save_ply(*pc.data(), options);
                }();
                if (!result)
                    throw_io_error(result.error(), "Failed to save point cloud PLY");
            },
            nb::arg("point_cloud"), nb::arg("path"), nb::arg("extra_attributes") = nb::none(),
            nb::arg("include_provenance") = true,
            "Save a point cloud as PLY file (xyz + colors) with optional extra per-vertex float attributes. "
            "include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.");

        m.def(
            "save_sog",
            [](const PySplatData& data, const std::filesystem::path& path, int kmeans_iterations, bool use_gpu,
               nb::object progress, bool include_provenance) {
                io::SogSaveOptions options;
                options.output_path = path;
                options.kmeans_iterations = kmeans_iterations;
                options.use_gpu = use_gpu;
                options.provenance = include_provenance ? core::make_provenance_stamp()
                                                        : core::make_minimal_provenance_stamp();

                if (progress && !progress.is_none()) {
                    PyExportProgressCallback py_progress{nb::cast<nb::object>(progress)};
                    options.progress_callback = [py_progress](float p, const std::string& stage) -> bool {
                        return py_progress(p, stage);
                    };
                }

                auto result = [&] {
                    nb::gil_scoped_release release;
                    return io::save_sog(*data.data(), options);
                }();
                if (!result)
                    throw_io_error(result.error(), "Failed to save SOG");
            },
            nb::arg("data"), nb::arg("path"), nb::arg("kmeans_iterations") = 10, nb::arg("use_gpu") = true,
            nb::arg("progress") = nb::none(),
            nb::arg("include_provenance") = true,
            "Save splat data as SOG compressed file. "
            "include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.");

        m.def(
            "save_ssog",
            [](const PySplatData& data, const std::filesystem::path& path, int lod_levels, float lod_ratio, int chunk_count_k, float chunk_extent,
               int chunk_min_k, int kmeans_iterations, bool use_gpu,
               nb::object progress, bool include_provenance) {
                io::SsogSaveOptions options;
                options.output_path = path;
                options.lod_levels = lod_levels;
                options.lod_ratio = lod_ratio;
                options.chunk_count_k = chunk_count_k;
                options.chunk_extent = chunk_extent;
                options.chunk_min_k = chunk_min_k;
                options.kmeans_iterations = kmeans_iterations;
                options.use_gpu = use_gpu;
                options.provenance = include_provenance ? core::make_provenance_stamp()
                                                        : core::make_minimal_provenance_stamp();

                if (progress && !progress.is_none()) {
                    PyExportProgressCallback py_progress{nb::cast<nb::object>(progress)};
                    options.progress_callback = [py_progress](float p, const std::string& stage) -> bool {
                        return py_progress(p, stage);
                    };
                }

                auto result = [&] {
                    nb::gil_scoped_release release;
                    return io::save_ssog(*data.data(), options);
                }();
                if (!result)
                    throw_io_error(result.error(), "Failed to save SSOG");
            },
            nb::arg("splat"), nb::arg("path"), nb::arg("lod_levels") = 4, nb::arg("lod_ratio") = 0.5f,
            nb::arg("chunk_count_k") = 512, nb::arg("chunk_extent") = 16.0f, nb::arg("chunk_min_k") = 8, nb::arg("kmeans_iterations") = 10, nb::arg("use_gpu") = true,
            nb::arg("progress") = nb::none(),
            nb::arg("include_provenance") = true,
            "Save splat data as a PlayCanvas multi-LOD SSOG (.ssog, lod-meta.json). "
            "include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.");

        m.def(
            "save_spz",
            [](const PySplatData& data, const std::filesystem::path& path, int version, bool include_provenance) {
                io::SpzSaveOptions options;
                options.output_path = path;
                options.version = version;
                options.provenance = include_provenance ? core::make_provenance_stamp()
                                                        : core::make_minimal_provenance_stamp();

                auto result = [&] {
                    nb::gil_scoped_release release;
                    return io::save_spz(*data.data(), options);
                }();
                if (!result)
                    throw_io_error(result.error(), "Failed to save SPZ");
            },
            nb::arg("data"), nb::arg("path"), nb::arg("version") = 4,
            nb::arg("include_provenance") = true,
            "Save splat data as SPZ compressed file.\n\n"
            "version: SPZ container version, 4 (zstd, default) or 3 (legacy gzip).\n"
            "include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded. Ignored for SPZ v3.");

        m.def(
            "save_usd",
            [](const PySplatData& data, const std::filesystem::path& path, bool include_provenance) {
                io::UsdSaveOptions options;
                options.output_path = path;
                options.provenance = include_provenance ? core::make_provenance_stamp()
                                                        : core::make_minimal_provenance_stamp();

                auto result = [&] {
                    nb::gil_scoped_release release;
                    return io::save_usd(*data.data(), options);
                }();
                if (!result)
                    throw_io_error(result.error(), "Failed to save USD");
            },
            nb::arg("data"), nb::arg("path"),
            nb::arg("include_provenance") = true,
            "Save splat data as OpenUSD gaussian file. "
            "include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.");

        m.def(
            "save_nurec_usdz",
            [](const PySplatData& data, const std::filesystem::path& path, bool include_provenance) {
                io::NurecUsdzSaveOptions options;
                options.output_path = path;
                options.provenance = include_provenance ? core::make_provenance_stamp()
                                                        : core::make_minimal_provenance_stamp();

                auto result = [&] {
                    nb::gil_scoped_release release;
                    return io::save_nurec_usdz(*data.data(), options);
                }();
                if (!result)
                    throw_io_error(result.error(), "Failed to save NuRec USDZ");
            },
            nb::arg("data"), nb::arg("path"),
            nb::arg("include_provenance") = true,
            "Save splat data as NuRec USDZ compatible with PLY_to_USD / Omniverse. "
            "include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.");

        m.def(
            "export_html",
            [](const PySplatData& data, const std::filesystem::path& path, int kmeans_iterations, nb::object progress,
               bool include_provenance) {
                io::HtmlExportOptions options;
                options.output_path = path;
                options.kmeans_iterations = kmeans_iterations;
                options.provenance = include_provenance ? core::make_provenance_stamp()
                                                        : core::make_minimal_provenance_stamp();

                if (progress && !progress.is_none()) {
                    PyExportProgressCallback py_progress{nb::cast<nb::object>(progress)};
                    options.progress_callback = [py_progress](float p, const std::string& stage) -> bool {
                        return py_progress(p, stage);
                    };
                }

                auto result = [&] {
                    nb::gil_scoped_release release;
                    return io::export_html(*data.data(), options);
                }();
                if (!result)
                    throw_io_error(result.error(), "Failed to export HTML");
            },
            nb::arg("data"), nb::arg("path"), nb::arg("kmeans_iterations") = 10, nb::arg("progress") = nb::none(),
            nb::arg("include_provenance") = true,
            "Export splat data as self-contained HTML viewer. "
            "include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.");

        m.def("is_ssog_path", &io::is_ssog_path, nb::arg("path"),
              "Check for an SSOG bundle, manifest or directory.");

        m.def(
            "is_dataset_path",
            [](const std::filesystem::path& path) {
                nb::gil_scoped_release release;
                return io::Loader::isDatasetPath(path);
            },
            nb::arg("path"),
            "Check if path is a dataset directory");

        m.def(
            "is_gaussian_splat_ply",
            [](const std::filesystem::path& path) { return io::is_gaussian_splat_ply(path); },
            nb::arg("path"),
            "Check if PLY file is a 3D Gaussian splat (has opacity, scale_0, rot_0 properties)");

        m.def(
            "get_supported_formats", []() {
                auto loader = io::Loader::create();
                return loader->getSupportedFormats();
            },
            "Get list of supported file format names");

        m.def(
            "get_supported_extensions", []() {
                auto loader = io::Loader::create();
                return loader->getSupportedExtensions();
            },
            "Get list of supported file extensions");

        m.def(
            "save_image",
            [](const std::filesystem::path& path, const PyTensor& image, bool include_provenance) {
                auto t = image.tensor().contiguous().cpu();
                const auto comment = core::provenance_to_json(
                    include_provenance ? core::make_provenance_stamp()
                                       : core::make_minimal_provenance_stamp());
                core::save_image(path, std::move(t), comment);
            },
            nb::arg("path"), nb::arg("image"),
            nb::arg("include_provenance") = true,
            "Save image tensor to file (PNG, JPG, TIFF, EXR). Accepts [H,W,C] or [C,H,W] float [0,1]. "
            "include_provenance (default true) writes a full Comment stamp on PNG and JPEG; when false, a minimal build stamp is still embedded.");
    }

} // namespace lfs::python
