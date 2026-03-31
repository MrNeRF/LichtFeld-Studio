/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/path_utils.hpp"
#include "py_tensor.hpp"
#include "training/dataset.hpp"
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>

namespace nb = nanobind;

namespace lfs::python {

    class PyCamera {
    public:
        explicit PyCamera(core::Camera* cam) : cam_(cam) {
            assert(cam_ != nullptr);
        }

        // Intrinsics
        float focal_x() const { return cam_->focal_x(); }
        float focal_y() const { return cam_->focal_y(); }
        float center_x() const { return cam_->center_x(); }
        float center_y() const { return cam_->center_y(); }
        float fov_x() const { return cam_->FoVx(); }
        float fov_y() const { return cam_->FoVy(); }

        // Image info
        int image_width() const { return cam_->image_width(); }
        int image_height() const { return cam_->image_height(); }
        int camera_width() const { return cam_->camera_width(); }
        int camera_height() const { return cam_->camera_height(); }
        std::string image_name() const { return cam_->image_name(); }
        std::string image_path() const { return lfs::core::path_to_utf8(cam_->image_path()); }
        std::string mask_path() const { return lfs::core::path_to_utf8(cam_->mask_path()); }
        bool has_mask() const { return cam_->has_mask(); }
        int uid() const { return cam_->uid(); }

        // Transforms (return PyTensor)
        PyTensor R() const { return PyTensor(cam_->R(), false); }
        PyTensor T() const { return PyTensor(cam_->T(), false); }
        PyTensor K() const { return PyTensor(cam_->K(), true); }
        PyTensor world_view_transform() const {
            return PyTensor(cam_->world_view_transform(), false);
        }
        PyTensor cam_position() const {
            return PyTensor(cam_->cam_position(), false);
        }

        // Load image/mask
        PyTensor load_image(int resize_factor = 1, int max_width = 3840) {
            return PyTensor(cam_->load_and_get_image(resize_factor, max_width), true);
        }
        PyTensor load_mask(int resize_factor = 1, int max_width = 3840,
                           bool invert = false, float threshold = 0.5f) {
            return PyTensor(cam_->load_and_get_mask(resize_factor, max_width, invert, threshold), true);
        }

        // Access underlying camera
        core::Camera* camera() { return cam_; }
        const core::Camera* camera() const { return cam_; }

    private:
        core::Camera* cam_;
    };

    class PyCameraDataset {
    public:
        explicit PyCameraDataset(std::shared_ptr<training::CameraDataset> dataset)
            : dataset_(std::move(dataset)) {
            assert(dataset_ != nullptr);
        }

        size_t size() const { return dataset_->size(); }

        PyCamera get(size_t index) {
            if (index >= dataset_->size()) {
                throw std::out_of_range("Camera index out of range");
            }
            auto example = dataset_->get(index);
            return PyCamera(example.data.camera);
        }

        std::optional<PyCamera> get_camera_by_filename(const std::string& filename) {
            auto cam_opt = dataset_->get_camera_by_filename(filename);
            if (!cam_opt)
                return std::nullopt;
            return PyCamera(*cam_opt);
        }

        std::vector<PyCamera> cameras() {
            std::vector<PyCamera> result;
            const auto& cams = dataset_->get_cameras();
            result.reserve(cams.size());
            for (const auto& cam : cams) {
                result.emplace_back(cam.get());
            }
            return result;
        }

        void set_resize_factor(int factor) { dataset_->set_resize_factor(factor); }
        void set_max_width(int width) { dataset_->set_max_width(width); }

        // Access underlying dataset
        std::shared_ptr<training::CameraDataset> dataset() { return dataset_; }

    private:
        std::shared_ptr<training::CameraDataset> dataset_;
    };

    // Register camera classes with nanobind module
    void register_cameras(nb::module_& m);

} // namespace lfs::python
