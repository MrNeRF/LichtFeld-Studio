/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_io.hpp"
#include "py_cameras.hpp"
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
#include "core/splat_data.hpp"
#include "formats/ply.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"
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
                    return PySplatData(splat_data.get());
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
                        std::format("Failed to load '{}': {}", path.string(), result.error().format()));
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
            [](const PySplatData& data, const std::filesystem::path& path, bool binary, nb::object progress) {
                io::PlySaveOptions options;
                options.output_path = path;
                options.binary = binary;

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
            "Save splat data as PLY file");

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
            "export_html",
            [](const PySplatData& data, const std::filesystem::path& path, int kmeans_iterations, nb::object progress) {
                io::HtmlExportOptions options;
                options.output_path = path;
                options.kmeans_iterations = kmeans_iterations;

                if (progress && !progress.is_none()) {
                    PyProgressCallback py_progress{nb::cast<nb::object>(progress)};
                    options.progress_callback = [py_progress](float p, const std::string& stage) {
                        py_progress(p, stage);
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
    }

} // namespace lfs::python
