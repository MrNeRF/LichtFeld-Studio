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
        float focal_x() const;
        float focal_y() const;
        float center_x() const;
        float center_y() const;
        float fov_x() const;
        float fov_y() const;

        // Image info
        int image_width() const;
        int image_height() const;
        int camera_width() const;
        int camera_height() const;
        std::string image_name() const;
        std::string image_path() const;
        std::string mask_path() const;
        bool has_mask() const;
        int uid() const;

        // Render/view contract: visualizer camera pose and derived view matrix.
        PyTensor rotation() const;
        PyTensor translation() const;
        PyTensor K() const;
        PyTensor view_matrix() const;

        // Deprecated raw dataset-camera properties kept as compatibility shims.
        PyTensor R() const;
        PyTensor T() const;
        PyTensor world_view_transform() const;
        PyTensor cam_position() const;

        // Load image/mask
        PyTensor load_image(int resize_factor = 1, int max_width = 0, bool output_uint8 = false);
        PyTensor load_mask(int resize_factor = 1, int max_width = 0,
                           bool invert = false, float threshold = 0.5f);

        // Access underlying camera
        core::Camera* camera();
        const core::Camera* camera() const;

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
            return PyCamera(dataset_->get_camera(index));
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
