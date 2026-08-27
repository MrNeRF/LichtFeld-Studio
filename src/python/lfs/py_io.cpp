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
#include "training/dataset.hpp"

#include <filesystem>
#include <format>
#include <optional>

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
                if (!callback)
                    return;
                nb::gil_scoped_acquire gil;
                try {
                    callback(progress, message);
                } catch (const std::exception& e) {
                    LOG_ERROR("Python progress callback error: {}", e.what());
                }
            }
        };

        struct PyExportProgressCallback {
            nb::object callback;

            bool operator()(float progress, const std::string& stage) const {
                if (!callback)
                    return true;
                nb::gil_scoped_acquire gil;
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
            .def_ro("fallback_preview_path", &PyProjectInspection::fallback_preview_path);

        m.def(
            "inspect_project",
            [](const std::filesystem::path& path) {
                project::ReaderOptions options;
                options.allow_unsupported_inspection = true;
                std::optional<lfs::Result<project::ProjectReader>> opened;
                std::string fallback_preview_path;
                {
                    nb::gil_scoped_release release;
                    opened = project::ProjectReader::open(path, options);
                    if (opened && opened->has_value() &&
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
                    .fallback_preview_path = std::move(fallback_preview_path),
                };
            },
            nb::arg("path"),
            "Inspect validated .licht container metadata without reading project payloads.");

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

                auto result = loader->load(path, options);
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
                const auto result = io::load_ply_point_cloud(path);
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

                auto result = io::save_ply(*data.data(), options);
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
                auto result = io::save_ply(*pc.data(), options);
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

                auto result = io::save_sog(*data.data(), options);
                if (!result)
                    throw_io_error(result.error(), "Failed to save SOG");
            },
            nb::arg("data"), nb::arg("path"), nb::arg("kmeans_iterations") = 10, nb::arg("use_gpu") = true,
            nb::arg("progress") = nb::none(),
            nb::arg("include_provenance") = true,
            "Save splat data as SOG compressed file. "
            "include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.");

        m.def(
            "save_spz",
            [](const PySplatData& data, const std::filesystem::path& path, int version, bool include_provenance) {
                io::SpzSaveOptions options;
                options.output_path = path;
                options.version = version;
                options.provenance = include_provenance ? core::make_provenance_stamp()
                                                        : core::make_minimal_provenance_stamp();

                auto result = io::save_spz(*data.data(), options);
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

                auto result = io::save_usd(*data.data(), options);
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

                auto result = io::save_nurec_usdz(*data.data(), options);
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

                auto result = io::export_html(*data.data(), options);
                if (!result)
                    throw_io_error(result.error(), "Failed to export HTML");
            },
            nb::arg("data"), nb::arg("path"), nb::arg("kmeans_iterations") = 10, nb::arg("progress") = nb::none(),
            nb::arg("include_provenance") = true,
            "Export splat data as self-contained HTML viewer. "
            "include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.");

        m.def(
            "is_dataset_path",
            [](const std::filesystem::path& path) { return io::Loader::isDatasetPath(path); },
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
