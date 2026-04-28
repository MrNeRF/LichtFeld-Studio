/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_cameras.hpp"
#include "rendering/coordinate_conventions.hpp"

#include "python_compat.hpp"
#include <cassert>
#include <format>
#include <numbers>

#include <glm/glm.hpp>

namespace {

    constexpr float RAD_TO_DEG = 180.0f / std::numbers::pi_v<float>;

    lfs::core::Tensor tensor_from_mat3_row_major(const glm::mat3& matrix) {
        auto tensor = lfs::core::Tensor::empty({3, 3}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto* ptr = static_cast<float*>(tensor.data_ptr());
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                ptr[row * 3 + col] = matrix[col][row];
            }
        }
        return tensor;
    }

    lfs::core::Tensor tensor_from_vec3(const glm::vec3& value) {
        return lfs::core::Tensor::from_vector({value.x, value.y, value.z}, {3}, lfs::core::Device::CPU);
    }

    lfs::core::Tensor tensor_from_mat4_row_major(const glm::mat4& matrix) {
        auto tensor = lfs::core::Tensor::empty({4, 4}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto* ptr = static_cast<float*>(tensor.data_ptr());
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                ptr[row * 4 + col] = matrix[col][row];
            }
        }
        return tensor;
    }

    glm::mat3 mat3_from_row_major_tensor(const lfs::core::Tensor& tensor) {
        const auto cpu = tensor.cpu().contiguous();
        return lfs::rendering::mat3FromRowMajor3x3(static_cast<const float*>(cpu.data_ptr()));
    }

    glm::vec3 vec3_from_tensor(const lfs::core::Tensor& tensor) {
        const auto cpu = tensor.cpu().contiguous();
        const auto* ptr = static_cast<const float*>(cpu.data_ptr());
        return {ptr[0], ptr[1], ptr[2]};
    }

    float radians_to_degrees(const float radians) {
        return radians * RAD_TO_DEG;
    }

    void warn_deprecated_camera_property(const std::string_view property,
                                         const std::string_view guidance) {
        const std::string message = std::format(
            "lichtfeld.scene.Camera.{} is deprecated; {}",
            property,
            guidance);
        if (PyErr_WarnEx(PyExc_DeprecationWarning, message.c_str(), 2) < 0) {
            throw nb::python_error();
        }
    }

    lfs::rendering::CameraPose visualizer_pose_from_camera(const lfs::core::Camera& camera) {
        return lfs::rendering::visualizerCameraPoseFromDataWorldToCamera(
            mat3_from_row_major_tensor(camera.R()),
            vec3_from_tensor(camera.T()));
    }

} // namespace

namespace lfs::python {

    float PyCamera::focal_x() const { return cam_->focal_x(); }
    float PyCamera::focal_y() const { return cam_->focal_y(); }
    float PyCamera::center_x() const { return cam_->center_x(); }
    float PyCamera::center_y() const { return cam_->center_y(); }
    float PyCamera::fov_x() const { return radians_to_degrees(cam_->FoVx()); }
    float PyCamera::fov_y() const { return radians_to_degrees(cam_->FoVy()); }

    int PyCamera::image_width() const { return cam_->image_width(); }
    int PyCamera::image_height() const { return cam_->image_height(); }
    int PyCamera::camera_width() const { return cam_->camera_width(); }
    int PyCamera::camera_height() const { return cam_->camera_height(); }
    std::string PyCamera::image_name() const { return cam_->image_name(); }
    std::string PyCamera::image_path() const { return lfs::core::path_to_utf8(cam_->image_path()); }
    std::string PyCamera::mask_path() const { return lfs::core::path_to_utf8(cam_->mask_path()); }
    bool PyCamera::has_mask() const { return cam_->has_mask(); }
    int PyCamera::uid() const { return cam_->uid(); }

    PyTensor PyCamera::rotation() const {
        return PyTensor(tensor_from_mat3_row_major(visualizer_pose_from_camera(*cam_).rotation).cuda(), true);
    }

    PyTensor PyCamera::translation() const {
        return PyTensor(tensor_from_vec3(visualizer_pose_from_camera(*cam_).translation).cuda(), true);
    }

    PyTensor PyCamera::K() const { return PyTensor(cam_->K(), true); }

    PyTensor PyCamera::view_matrix() const {
        const auto pose = visualizer_pose_from_camera(*cam_);
        return PyTensor(tensor_from_mat4_row_major(
                            lfs::rendering::makeViewMatrix(pose.rotation, pose.translation))
                            .cuda(),
                        true);
    }

    PyTensor PyCamera::R() const {
        warn_deprecated_camera_property(
            "R",
            "it exposes the raw dataset world-to-camera rotation; use lichtfeld.scene.Camera.rotation for the visualizer pose or lichtfeld.scene.Camera.view_matrix for a render-space view matrix");
        return PyTensor(cam_->R(), false);
    }

    PyTensor PyCamera::T() const {
        warn_deprecated_camera_property(
            "T",
            "it exposes the raw dataset world-to-camera translation; use lichtfeld.scene.Camera.translation for the visualizer pose or lichtfeld.scene.Camera.view_matrix for a render-space view matrix");
        return PyTensor(cam_->T(), false);
    }

    PyTensor PyCamera::world_view_transform() const {
        warn_deprecated_camera_property(
            "world_view_transform",
            "it exposes the raw dataset world-to-camera transform; use lichtfeld.scene.Camera.view_matrix for the visualizer render contract");
        return PyTensor(cam_->world_view_transform(), false);
    }

    PyTensor PyCamera::cam_position() const {
        warn_deprecated_camera_property(
            "cam_position",
            "it exposes the raw dataset-world camera position; use lichtfeld.scene.Camera.translation for the visualizer-space camera position");
        return PyTensor(cam_->cam_position(), false);
    }

    PyTensor PyCamera::load_image(int resize_factor, int max_width, const bool output_uint8) {
        return PyTensor(cam_->load_and_get_image(resize_factor, max_width, output_uint8), true);
    }

    PyTensor PyCamera::load_mask(int resize_factor, int max_width, bool invert, float threshold) {
        return PyTensor(cam_->load_and_get_mask(resize_factor, max_width, invert, threshold), true);
    }

    core::Camera* PyCamera::camera() { return cam_; }
    const core::Camera* PyCamera::camera() const { return cam_; }

    void register_cameras(nb::module_& m) {
        // Camera class
        nb::class_<PyCamera>(m, "Camera")
            // Intrinsics
            .def_prop_ro("focal_x", &PyCamera::focal_x, "Focal length X in pixels")
            .def_prop_ro("focal_y", &PyCamera::focal_y, "Focal length Y in pixels")
            .def_prop_ro("center_x", &PyCamera::center_x, "Principal point X in pixels")
            .def_prop_ro("center_y", &PyCamera::center_y, "Principal point Y in pixels")
            .def_prop_ro("fov_x", &PyCamera::fov_x, "Horizontal field of view in degrees")
            .def_prop_ro("fov_y", &PyCamera::fov_y, "Vertical field of view in degrees")
            // Image info
            .def_prop_ro("image_width", &PyCamera::image_width, "Image width in pixels")
            .def_prop_ro("image_height", &PyCamera::image_height, "Image height in pixels")
            .def_prop_ro("camera_width", &PyCamera::camera_width, "Camera sensor width")
            .def_prop_ro("camera_height", &PyCamera::camera_height, "Camera sensor height")
            .def_prop_ro("image_name", &PyCamera::image_name, "Image filename")
            .def_prop_ro("image_path", &PyCamera::image_path, "Full path to image file")
            .def_prop_ro("mask_path", &PyCamera::mask_path, "Full path to mask file")
            .def_prop_ro("has_mask", &PyCamera::has_mask, "Whether a mask file exists")
            .def_prop_ro("uid", &PyCamera::uid, "Unique camera identifier")
            // Visualizer-space camera pose, directly compatible with render_view().
            .def_prop_ro("rotation", &PyCamera::rotation,
                         "Visualizer camera-to-world rotation [3, 3], directly usable with render_view()")
            .def_prop_ro("translation", &PyCamera::translation,
                         "Visualizer camera position [3], directly usable with render_view()")
            .def_prop_ro("K", &PyCamera::K, "Intrinsic matrix [3, 3]")
            .def_prop_ro("view_matrix", &PyCamera::view_matrix,
                         "Visualizer world-to-camera view matrix [4, 4]")
            .def_prop_ro("R", &PyCamera::R,
                         "Deprecated raw dataset world-to-camera rotation [3, 3]")
            .def_prop_ro("T", &PyCamera::T,
                         "Deprecated raw dataset world-to-camera translation [3]")
            .def_prop_ro("world_view_transform", &PyCamera::world_view_transform,
                         "Deprecated raw dataset world-to-camera transform [1, 4, 4]")
            .def_prop_ro("cam_position", &PyCamera::cam_position,
                         "Deprecated raw dataset-world camera position [3]")
            // Load methods
            .def("load_image", &PyCamera::load_image,
                 nb::arg("resize_factor") = 1, nb::arg("max_width") = 0,
                 nb::arg("output_uint8") = false,
                 "Load image as tensor [C, H, W] on CUDA. Set output_uint8=True to return uint8 [0,255] instead of float32 [0,1].")
            .def("load_mask", &PyCamera::load_mask,
                 nb::arg("resize_factor") = 1, nb::arg("max_width") = 0,
                 nb::arg("invert") = false, nb::arg("threshold") = 0.5f,
                 "Load mask as tensor [1, H, W] on CUDA");

        // CameraDataset class
        nb::class_<PyCameraDataset>(m, "CameraDataset")
            .def("__len__", &PyCameraDataset::size, "Number of cameras")
            .def("__getitem__", &PyCameraDataset::get, nb::arg("index"), "Get camera by index")
            .def("get_camera_by_filename", &PyCameraDataset::get_camera_by_filename,
                 nb::arg("filename"), "Find camera by image filename")
            .def("cameras", &PyCameraDataset::cameras,
                 "Get all cameras as a list")
            .def("set_resize_factor", &PyCameraDataset::set_resize_factor,
                 nb::arg("factor"), "Set image resize factor for all cameras")
            .def("set_max_width", &PyCameraDataset::set_max_width,
                 nb::arg("width"), "Set maximum image width for all cameras");
    }

} // namespace lfs::python
