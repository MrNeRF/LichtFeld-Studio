/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_types.h"
#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <future>
#include <string>

namespace lfs::core {

    class Camera {
    public:
        Camera() = default;

        Camera(const Tensor& R,
               const Tensor& T,
               float focal_x, float focal_y,
               float center_x, float center_y,
               Tensor radial_distortion,
               Tensor tangential_distortion,
               CameraModelType camera_model_type,
               const std::string& image_name,
               const std::filesystem::path& image_path,
               const std::filesystem::path& mask_path,
               int camera_width, int camera_height,
               int uid);
        Camera(const Camera&, const Tensor& transform);

        // Destructor to clean up CUDA stream
        ~Camera();

        // Delete copy, define proper move semantics
        Camera(const Camera&) = delete;
        Camera& operator=(const Camera&) = delete;
        Camera(Camera&& other) noexcept;
        Camera& operator=(Camera&& other) noexcept;

        // Initialize GPU tensors on demand
        void initialize_cuda_tensors();

        // Load image from disk and return it
        Tensor load_and_get_image(int resize_factor = -1, int max_width = 3840);

        // Load mask from disk, process it, and return it (cached)
        Tensor load_and_get_mask(int resize_factor = -1, int max_width = 3840,
                                 bool invert_mask = false, float mask_threshold = 0.5f);

        // Load image from disk just to populate _image_width/_image_height
        void load_image_size(int resize_factor = -1, int max_width = 3840);

        // Get number of bytes in the image file
        size_t get_num_bytes_from_file(int resize_factor = -1, int max_width = 3840) const;
        size_t get_num_bytes_from_file() const;

        // Accessors - now return const references to avoid copies
        const Tensor& world_view_transform() const {
            return _world_view_transform;
        }
        const Tensor& cam_position() const {
            return _cam_position;
        }

        // Direct GPU pointer access (tensors are already contiguous on CUDA)
        const float* world_view_transform_ptr() const {
            return _world_view_transform.ptr<float>();
        }
        const float* cam_position_ptr() const {
            return _cam_position.ptr<float>();
        }

        const Tensor& R() const { return _R; }
        const Tensor& T() const { return _T; }

        Tensor K() const;

        std::tuple<float, float, float, float> get_intrinsics() const;

        int image_height() const noexcept { return _image_height; }
        int image_width() const noexcept { return _image_width; }
        void set_image_dimensions(int width, int height) noexcept {
            _image_width = width;
            _image_height = height;
        }
        int camera_height() const noexcept { return _camera_height; }
        int camera_width() const noexcept { return _camera_width; }
        float focal_x() const noexcept { return _focal_x; }
        float focal_y() const noexcept { return _focal_y; }
        float center_x() const noexcept { return _center_x; }
        float center_y() const noexcept { return _center_y; }
        Tensor radial_distortion() const noexcept { return _radial_distortion; }
        Tensor tangential_distortion() const noexcept { return _tangential_distortion; }
        CameraModelType camera_model_type() const noexcept { return _camera_model_type; }
        const std::string& image_name() const noexcept { return _image_name; }
        const std::filesystem::path& image_path() const noexcept { return _image_path; }
        const std::filesystem::path& mask_path() const noexcept { return _mask_path; }
        bool has_mask() const noexcept { return !_mask_path.empty() && std::filesystem::exists(_mask_path); }
        int uid() const noexcept { return _uid; }

        float FoVx() const noexcept { return _FoVx; }
        float FoVy() const noexcept { return _FoVy; }

    private:
        // IDs
        float _FoVx = 0.f;
        float _FoVy = 0.f;
        int _uid = -1;
        float _focal_x = 0.f;
        float _focal_y = 0.f;
        float _center_x = 0.f;
        float _center_y = 0.f;

        // redundancy with _world_view_transform, but save calculation and passing from GPU 2 CPU
        Tensor _R;
        Tensor _T;

        Tensor _radial_distortion;
        Tensor _tangential_distortion;
        CameraModelType _camera_model_type = CameraModelType::PINHOLE;

        // Image info
        std::string _image_name;
        std::filesystem::path _image_path;
        std::filesystem::path _mask_path;
        int _camera_width = 0;
        int _camera_height = 0;
        int _image_width = 0;
        int _image_height = 0;

        // GPU tensors (computed on demand)
        Tensor _world_view_transform;
        Tensor _cam_position;

        // Mask caching (processed mask stored on GPU)
        Tensor _cached_mask;
        bool _mask_loaded = false;

        // CUDA stream for async operations
        cudaStream_t _stream = nullptr;
    };
    inline float focal2fov(float focal, int pixels) {
        return 2.0f * std::atan(pixels / (2.0f * focal));
    }

    inline float fov2focal(float fov, int pixels) {
        float tan_fov = std::tan(fov * 0.5f);
        return pixels / (2.0f * tan_fov);
    }

} // namespace lfs::core
