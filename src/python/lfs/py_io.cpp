/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_io.hpp"
#include "py_cameras.hpp"
#include "py_rml.hpp"
#include "py_scene.hpp"
#include "py_splat_data.hpp"
#include "py_tensor.hpp"

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "gui/rmlui/elements/video_preview_element.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"
#include "io/ply_export_internal.hpp"
#include "python_runtime.hpp"
#include "training/dataset.hpp"

#include <filesystem>
#include <format>

namespace lfs::python {

    namespace {

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

        core::Tensor tensor_from_python_attribute(const nb::handle& value) {
            if (nb::isinstance<PyTensor>(value)) {
                return nb::cast<PyTensor>(value).tensor();
            }

            if (nb::isinstance<nb::ndarray<>>(value)) {
                return PyTensor::from_numpy(nb::cast<nb::ndarray<>>(value)).tensor();
            }

            throw std::runtime_error(
                "extra_attributes values must be lichtfeld.Tensor or numpy.ndarray");
        }

        std::vector<io::PlyAttributeBlock> parse_extra_ply_attributes(const nb::object& extra_attributes,
                                                                      const std::filesystem::path& output_path) {
            if (!extra_attributes || extra_attributes.is_none()) {
                return {};
            }

            if (!nb::isinstance<nb::dict>(extra_attributes)) {
                throw std::runtime_error(
                    "extra_attributes must be a dict[str, lichtfeld.Tensor | numpy.ndarray]");
            }

            nb::dict attributes = nb::cast<nb::dict>(extra_attributes);
            std::vector<io::PlyAttributeBlock> blocks;
            blocks.reserve(attributes.size());

            for (const auto& item : attributes) {
                const std::string name = nb::cast<std::string>(item.first);
                if (name.empty()) {
                    throw std::runtime_error("extra_attributes keys must not be empty");
                }

                auto values = tensor_from_python_attribute(item.second);
                if (!values.is_valid() || values.numel() == 0) {
                    throw std::runtime_error(std::format(
                        "extra_attributes['{}'] must not be empty", name));
                }

                if (values.ndim() != 1 && values.ndim() != 2) {
                    throw std::runtime_error(std::format(
                        "extra_attributes['{}'] must be shaped [N] or [N,C]", name));
                }

                const size_t cols = values.ndim() == 1 ? 1 : static_cast<size_t>(values.size(1));
                if (cols == 0) {
                    throw std::runtime_error(std::format(
                        "extra_attributes['{}'] must have at least one column", name));
                }

                auto names = io::make_ply_extra_attribute_names(name, cols);
                if (auto result = io::validate_reserved_ply_extra_attribute_names(names, output_path); !result) {
                    throw std::runtime_error(result.error().format());
                }

                blocks.push_back(io::PlyAttributeBlock{
                    .values = std::move(values),
                    .names = std::move(names),
                });
            }

            return blocks;
        }

    } // namespace

    void register_io(nb::module_& m) {
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
               std::optional<std::string> images_folder, nb::object progress) -> PyLoadResult {
                auto loader = io::Loader::create();

                io::LoadOptions options;
                if (resize_factor)
                    options.resize_factor = *resize_factor;
                if (max_width)
                    options.max_width = *max_width;
                if (images_folder)
                    options.images_folder = *images_folder;

                if (progress && !progress.is_none()) {
                    PyProgressCallback py_progress{nb::cast<nb::object>(progress)};
                    options.progress = [py_progress](float p, const std::string& msg) {
                        py_progress(p, msg);
                    };
                }

                auto result = loader->load(path, options);
                if (!result) {
                    throw std::runtime_error(
                        std::format("Failed to load '{}': {}", lfs::core::path_to_utf8(path), result.error().format()));
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
            "Load a scene or splat file from path");

        m.def(
            "load_point_cloud",
            [](const std::filesystem::path& path) -> nb::tuple {
                const auto result = io::load_ply_point_cloud(path);
                if (!result)
                    throw std::runtime_error(std::format("Failed to load point cloud: {}", result.error()));
                return nb::make_tuple(PyTensor(result->means, true), PyTensor(result->colors, true));
            },
            nb::arg("path"),
            "Load a PLY as point cloud, returns (means [N,3], colors [N,3]) tensors");

        m.def(
            "save_ply",
            [](const PySplatData& data, const std::filesystem::path& path, bool binary,
               nb::object progress, nb::object extra_attributes) {
                io::PlySaveOptions options;
                options.output_path = path;
                options.binary = binary;
                options.extra_attributes = parse_extra_ply_attributes(extra_attributes, path);

                if (progress && !progress.is_none()) {
                    PyExportProgressCallback py_progress{nb::cast<nb::object>(progress)};
                    options.progress_callback = [py_progress](float p, const std::string& stage) -> bool {
                        return py_progress(p, stage);
                    };
                }

                auto result = io::save_ply(*data.data(), options);
                if (!result)
                    throw std::runtime_error(std::format("Failed to save PLY: {}", result.error().format()));
            },
            nb::arg("data"), nb::arg("path"), nb::arg("binary") = true, nb::arg("progress") = nb::none(),
            nb::arg("extra_attributes") = nb::none(),
            "Save splat data as PLY file with optional extra per-vertex float attributes");

        m.def(
            "save_point_cloud_ply",
            [](const PyPointCloud& pc, const std::filesystem::path& path, nb::object extra_attributes) {
                if (!pc.data())
                    throw std::runtime_error("Point cloud data must not be null");
                io::PlySaveOptions options;
                options.output_path = path;
                options.binary = true;
                options.extra_attributes = parse_extra_ply_attributes(extra_attributes, path);
                auto result = io::save_ply(*pc.data(), options);
                if (!result)
                    throw std::runtime_error(std::format("Failed to save point cloud PLY: {}", result.error().format()));
            },
            nb::arg("point_cloud"), nb::arg("path"), nb::arg("extra_attributes") = nb::none(),
            "Save a point cloud as PLY file (xyz + colors) with optional extra per-vertex float attributes");

        m.def(
            "save_sog",
            [](const PySplatData& data, const std::filesystem::path& path, int kmeans_iterations, bool use_gpu,
               nb::object progress) {
                io::SogSaveOptions options;
                options.output_path = path;
                options.kmeans_iterations = kmeans_iterations;
                options.use_gpu = use_gpu;

                if (progress && !progress.is_none()) {
                    PyExportProgressCallback py_progress{nb::cast<nb::object>(progress)};
                    options.progress_callback = [py_progress](float p, const std::string& stage) -> bool {
                        return py_progress(p, stage);
                    };
                }

                auto result = io::save_sog(*data.data(), options);
                if (!result)
                    throw std::runtime_error(std::format("Failed to save SOG: {}", result.error().format()));
            },
            nb::arg("data"), nb::arg("path"), nb::arg("kmeans_iterations") = 10, nb::arg("use_gpu") = true,
            nb::arg("progress") = nb::none(),
            "Save splat data as SOG compressed file");

        m.def(
            "save_spz",
            [](const PySplatData& data, const std::filesystem::path& path) {
                io::SpzSaveOptions options;
                options.output_path = path;

                auto result = io::save_spz(*data.data(), options);
                if (!result)
                    throw std::runtime_error(std::format("Failed to save SPZ: {}", result.error().format()));
            },
            nb::arg("data"), nb::arg("path"),
            "Save splat data as SPZ compressed file");

        m.def(
            "save_usd",
            [](const PySplatData& data, const std::filesystem::path& path) {
                io::UsdSaveOptions options;
                options.output_path = path;

                auto result = io::save_usd(*data.data(), options);
                if (!result)
                    throw std::runtime_error(std::format("Failed to save USD: {}", result.error().format()));
            },
            nb::arg("data"), nb::arg("path"),
            "Save splat data as OpenUSD gaussian file");

        m.def(
            "save_nurec_usdz",
            [](const PySplatData& data, const std::filesystem::path& path) {
                io::NurecUsdzSaveOptions options;
                options.output_path = path;

                auto result = io::save_nurec_usdz(*data.data(), options);
                if (!result)
                    throw std::runtime_error(std::format("Failed to save NuRec USDZ: {}", result.error().format()));
            },
            nb::arg("data"), nb::arg("path"),
            "Save splat data as NuRec USDZ compatible with PLY_to_USD / Omniverse");

        m.def(
            "export_html",
            [](const PySplatData& data, const std::filesystem::path& path, int kmeans_iterations, nb::object progress) {
                io::HtmlExportOptions options;
                options.output_path = path;
                options.kmeans_iterations = kmeans_iterations;

                if (progress && !progress.is_none()) {
                    PyExportProgressCallback py_progress{nb::cast<nb::object>(progress)};
                    options.progress_callback = [py_progress](float p, const std::string& stage) -> bool {
                        return py_progress(p, stage);
                    };
                }

                auto result = io::export_html(*data.data(), options);
                if (!result)
                    throw std::runtime_error(std::format("Failed to export HTML: {}", result.error().format()));
            },
            nb::arg("data"), nb::arg("path"), nb::arg("kmeans_iterations") = 10, nb::arg("progress") = nb::none(),
            "Export splat data as self-contained HTML viewer");

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
            [](const std::filesystem::path& path, const PyTensor& image) {
                auto t = image.tensor().contiguous().cpu();
                core::save_image(path, std::move(t));
            },
            nb::arg("path"), nb::arg("image"),
            "Save image tensor to file (PNG, JPG, TIFF, EXR). Accepts [H,W,C] or [C,H,W] float [0,1].");

        // ── Video player bindings (callback-based, no lfs_video link) ──

        m.def(
            "video_player_open",
            [](const std::string& path) {
                const auto& cbs = python::get_video_player_callbacks();
                if (!cbs.open)
                    throw std::runtime_error("Video player not available");
                if (!cbs.open(path))
                    throw std::runtime_error("Failed to open video: " + path);
            },
            nb::arg("path"), "Open a video file for playback");

        m.def(
            "video_player_close",
            [] {
                const auto& cbs = python::get_video_player_callbacks();
                if (cbs.close)
                    cbs.close();
            },
            "Close the current video");

        m.def(
            "video_player_is_open",
            [] {
                const auto& cbs = python::get_video_player_callbacks();
                return cbs.is_open ? cbs.is_open() : false;
            },
            "Check if a video is currently open");

        m.def(
            "video_player_play", [] {
                const auto& cbs = python::get_video_player_callbacks();
                if (cbs.play) cbs.play();
            },
            "Start video playback");

        m.def(
            "video_player_pause", [] {
                const auto& cbs = python::get_video_player_callbacks();
                if (cbs.pause) cbs.pause();
            },
            "Pause video playback");

        m.def(
            "video_player_toggle_play_pause", [] {
                const auto& cbs = python::get_video_player_callbacks();
                if (cbs.toggle_play_pause) cbs.toggle_play_pause();
            },
            "Toggle play/pause");

        m.def(
            "video_player_is_playing",
            [] {
                const auto& cbs = python::get_video_player_callbacks();
                return cbs.is_playing ? cbs.is_playing() : false;
            },
            "Check if video is currently playing");

        m.def(
            "video_player_seek",
            [](double seconds) {
                const auto& cbs = python::get_video_player_callbacks();
                if (cbs.seek) cbs.seek(seconds);
            },
            nb::arg("seconds"), "Seek to a specific time");

        m.def(
            "video_player_step_forward", [] {
                const auto& cbs = python::get_video_player_callbacks();
                if (cbs.step_forward) cbs.step_forward();
            },
            "Step forward one frame");

        m.def(
            "video_player_step_backward", [] {
                const auto& cbs = python::get_video_player_callbacks();
                if (cbs.step_backward) cbs.step_backward();
            },
            "Step backward one frame");

        m.def(
            "video_player_update",
            [] {
                const auto& cbs = python::get_video_player_callbacks();
                return cbs.update ? cbs.update() : false;
            },
            "Update video playback, returns True if frame changed");

        m.def(
            "video_player_current_time",
            [] {
                const auto& cbs = python::get_video_player_callbacks();
                return cbs.current_time ? cbs.current_time() : 0.0;
            },
            "Get current playback time in seconds");

        m.def(
            "video_player_duration",
            [] {
                const auto& cbs = python::get_video_player_callbacks();
                return cbs.duration ? cbs.duration() : 0.0;
            },
            "Get video duration in seconds");

        m.def(
            "video_player_fps",
            [] {
                const auto& cbs = python::get_video_player_callbacks();
                return cbs.fps ? cbs.fps() : 0.0;
            },
            "Get video FPS");

        m.def(
            "video_player_total_frames",
            [] {
                const auto& cbs = python::get_video_player_callbacks();
                return cbs.total_frames ? cbs.total_frames() : int64_t{0};
            },
            "Get total frame count");

        // ── Feed preview element ──

        m.def(
            "feed_preview_element",
            [](PyRmlElement& py_elem) {
                auto* raw = py_elem.raw();
                auto* preview = dynamic_cast<lfs::vis::gui::VideoPreviewElement*>(raw);
                if (!preview)
                    return;

                const auto& cbs = python::get_video_player_callbacks();
                if (!cbs.current_frame_data || !cbs.width || !cbs.height)
                    return;

                const uint8_t* data = cbs.current_frame_data();
                if (!data)
                    return;

                const int w = cbs.width();
                const int h = cbs.height();
                preview->setFrameData(data, w, h);
            },
            nb::arg("element"),
            "Feed current video frame to a VideoPreviewElement");

        // ── Video frame extraction ──

        m.def(
            "video_extract_frames",
            [](const std::string& video_path, const std::string& output_dir,
               int mode, double fps, int frame_interval, int format,
               int jpg_quality, double start_time, double end_time,
               int resolution_mode, float scale, int custom_width,
               int custom_height, const std::string& filename_pattern) {
                python::VideoExtractParams params;
                params.video_path = video_path;
                params.output_dir = output_dir;
                params.mode = mode;
                params.fps = fps;
                params.frame_interval = frame_interval;
                params.format = format;
                params.jpg_quality = jpg_quality;
                params.start_time = start_time;
                params.end_time = end_time;
                params.resolution_mode = resolution_mode;
                params.scale = scale;
                params.custom_width = custom_width;
                params.custom_height = custom_height;
                params.filename_pattern = filename_pattern;
                python::invoke_video_extract(params);
            },
            nb::arg("video_path"), nb::arg("output_dir"),
            nb::arg("mode") = 0, nb::arg("fps") = 1.0,
            nb::arg("frame_interval") = 1, nb::arg("format") = 0,
            nb::arg("jpg_quality") = 95, nb::arg("start_time") = 0.0,
            nb::arg("end_time") = -1.0, nb::arg("resolution_mode") = 0,
            nb::arg("scale") = 1.0f, nb::arg("custom_width") = 0,
            nb::arg("custom_height") = 0,
            nb::arg("filename_pattern") = "frame_%d",
            "Extract frames from a video file");

        m.def(
            "video_extract_state",
            [] {
                auto state = python::get_video_extract_state();
                nb::dict result;
                result["active"] = state.active;
                result["current"] = state.current;
                result["total"] = state.total;
                result["error"] = state.error;
                return result;
            },
            "Get current video extraction state");
    }

} // namespace lfs::python
