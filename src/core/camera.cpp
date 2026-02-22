/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "core/logger.hpp"
#include <cassert>
#include <cuda_runtime.h>

namespace lfs::core {
    static Tensor world_to_view(const Tensor& R, const Tensor& t) {
        // Create 4x4 identity matrix
        auto w2c = Tensor::eye(4, R.device());

        // Set rotation and translation parts
        auto w2c_cpu = w2c.cpu();
        auto R_cpu = R.cpu();
        auto t_cpu = t.cpu();

        auto w2c_acc = w2c_cpu.accessor<float, 2>();
        auto R_acc = R_cpu.accessor<float, 2>();
        auto t_acc = t_cpu.accessor<float, 1>();

        // Copy rotation [0:3, 0:3] = R
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                w2c_acc(i, j) = R_acc(i, j);
            }
        }

        // Copy translation [0:3, 3] = t
        for (size_t i = 0; i < 3; ++i) {
            w2c_acc(i, 3) = t_acc(i);
        }

        // Return as [1, 4, 4] on CUDA
        return w2c_cpu.to(Device::CUDA).unsqueeze(0).contiguous();
    }

    Camera::Camera(const Tensor& R,
                   const Tensor& T,
                   float focal_x, float focal_y,
                   float center_x, float center_y,
                   const Tensor radial_distortion,
                   const Tensor tangential_distortion,
                   CameraModelType camera_model_type,
                   const std::string& image_name,
                   const std::filesystem::path& image_path,
                   const std::filesystem::path& mask_path,
                   int camera_width, int camera_height,
                   int uid,
                   int camera_id)
        : _uid(uid),
          _camera_id(camera_id),
          _focal_x(focal_x),
          _focal_y(focal_y),
          _center_x(center_x),
          _center_y(center_y),
          _R(R),
          _T(T),
          _radial_distortion(radial_distortion),
          _tangential_distortion(tangential_distortion),
          _camera_model_type(camera_model_type),
          _image_name(image_name),
          _image_path(image_path),
          _mask_path(mask_path),
          _camera_width(camera_width),
          _camera_height(camera_height),
          _image_width(camera_width),
          _image_height(camera_height) {

        // Validate inputs
        if (!R.is_valid() || R.numel() == 0) {
            LOG_ERROR("Camera constructor: R tensor is invalid or empty");
            throw std::runtime_error("Camera constructor: R tensor is invalid or empty");
        }
        if (!T.is_valid() || T.numel() == 0) {
            LOG_ERROR("Camera constructor: T tensor is invalid or empty");
            throw std::runtime_error("Camera constructor: T tensor is invalid or empty");
        }

        // Compute world-to-view transform
        _world_view_transform = world_to_view(R, T);

        // Compute camera position: inverse of w2v gives c2w, position is c2w[:3, 3]
        // For transformation matrix [R|t], inverse is [R^T | -R^T*t]
        auto w2v_cpu = _world_view_transform.squeeze(0).cpu();
        auto w2v_acc = w2v_cpu.accessor<float, 2>();

        // Extract 3x3 rotation part
        std::vector<float> rot_data(9);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                rot_data[i * 3 + j] = w2v_acc(i, j);
            }
        }
        auto R_part = Tensor::from_vector(rot_data, TensorShape({3, 3}), Device::CPU);

        // Extract translation part
        std::vector<float> t_data = {w2v_acc(0, 3), w2v_acc(1, 3), w2v_acc(2, 3)};
        auto t_part = Tensor::from_vector(t_data, TensorShape({3}), Device::CPU);

        // Compute camera position: -R^T * t
        auto R_T = R_part.transpose(0, 1);
        auto cam_pos = R_T.matmul(t_part.unsqueeze(1)).squeeze(1).neg();

        _cam_position = cam_pos.to(Device::CUDA).contiguous();

        _FoVx = focal2fov(_focal_x, _camera_width);
        _FoVy = focal2fov(_focal_y, _camera_height);

        // Create CUDA stream for async image loading (like reference implementation)
        cudaStreamCreate(&_stream);
    }

    Camera::~Camera() {
        // Destroy CUDA stream if it was created
        if (_stream) {
            cudaStreamDestroy(_stream);
            _stream = nullptr;
        }
    }

    Camera::Camera(Camera&& other) noexcept
        : _FoVx(other._FoVx),
          _FoVy(other._FoVy),
          _uid(other._uid),
          _camera_id(other._camera_id),
          _focal_x(other._focal_x),
          _focal_y(other._focal_y),
          _center_x(other._center_x),
          _center_y(other._center_y),
          _R(std::move(other._R)),
          _T(std::move(other._T)),
          _radial_distortion(std::move(other._radial_distortion)),
          _tangential_distortion(std::move(other._tangential_distortion)),
          _image_path(std::move(other._image_path)),
          _image_name(std::move(other._image_name)),
          _mask_path(std::move(other._mask_path)),
          _camera_width(other._camera_width),
          _camera_height(other._camera_height),
          _image_width(other._image_width),
          _image_height(other._image_height),
          _world_view_transform(std::move(other._world_view_transform)),
          _cam_position(std::move(other._cam_position)),
          _cached_mask(std::move(other._cached_mask)),
          _mask_loaded(other._mask_loaded),
          _undistort_precomputed(other._undistort_precomputed),
          _undistort_prepared(other._undistort_prepared),
          _undistort_params(other._undistort_params),
          _stream(other._stream) {
        // Take ownership of the stream
        other._stream = nullptr;
        other._mask_loaded = false;
        other._undistort_precomputed = false;
        other._undistort_prepared = false;
    }

    Camera& Camera::operator=(Camera&& other) noexcept {
        if (this != &other) {
            // Destroy our current stream
            if (_stream) {
                cudaStreamDestroy(_stream);
            }

            // Move all members
            _FoVx = other._FoVx;
            _FoVy = other._FoVy;
            _uid = other._uid;
            _camera_id = other._camera_id;
            _focal_x = other._focal_x;
            _focal_y = other._focal_y;
            _center_x = other._center_x;
            _center_y = other._center_y;
            _R = std::move(other._R);
            _T = std::move(other._T);
            _radial_distortion = std::move(other._radial_distortion);
            _tangential_distortion = std::move(other._tangential_distortion);
            _image_path = std::move(other._image_path);
            _image_name = std::move(other._image_name);
            _mask_path = std::move(other._mask_path);
            _camera_width = other._camera_width;
            _camera_height = other._camera_height;
            _image_width = other._image_width;
            _image_height = other._image_height;
            _world_view_transform = std::move(other._world_view_transform);
            _cam_position = std::move(other._cam_position);
            _cached_mask = std::move(other._cached_mask);
            _mask_loaded = other._mask_loaded;
            _undistort_precomputed = other._undistort_precomputed;
            _undistort_prepared = other._undistort_prepared;
            _undistort_params = other._undistort_params;

            // Take ownership of the stream
            _stream = other._stream;
            other._stream = nullptr;
            other._mask_loaded = false;
            other._undistort_precomputed = false;
            other._undistort_prepared = false;
        }
        return *this;
    }

    Camera::Camera(const Camera& other, const Tensor& transform)
        : _uid(other._uid),
          _camera_id(other._camera_id),
          _focal_x(other._focal_x),
          _focal_y(other._focal_y),
          _center_x(other._center_x),
          _center_y(other._center_y),
          _R(other._R),
          _T(other._T),
          _radial_distortion(other._radial_distortion),
          _tangential_distortion(other._tangential_distortion),
          _camera_model_type(other._camera_model_type),
          _image_name(other._image_name),
          _image_path(other._image_path),
          _mask_path(other._mask_path),
          _camera_width(other._camera_width),
          _camera_height(other._camera_height),
          _image_width(other._image_width),
          _image_height(other._image_height),
          _cam_position(other._cam_position),
          _FoVx(other._FoVx),
          _FoVy(other._FoVy) {
        _world_view_transform = transform;

        // Create CUDA stream for async image loading
        cudaStreamCreate(&_stream);
    }
    Tensor Camera::K() const {
        // Create [1, 3, 3] zero matrix on same device as world_view_transform
        auto K = Tensor::zeros({1, 3, 3}, _world_view_transform.device());
        auto [fx, fy, cx, cy] = get_intrinsics();

        // Fill in the intrinsic matrix on CPU then move to device
        auto K_cpu = K.cpu();
        auto K_acc = K_cpu.accessor<float, 3>();
        K_acc(0, 0, 0) = fx;
        K_acc(0, 1, 1) = fy;
        K_acc(0, 0, 2) = cx;
        K_acc(0, 1, 2) = cy;
        K_acc(0, 2, 2) = 1.0f;

        return K_cpu.to(_world_view_transform.device()).contiguous();
    }

    std::tuple<float, float, float, float> Camera::get_intrinsics() const {
        const float x_scale = static_cast<float>(_image_width) / static_cast<float>(_camera_width);
        const float y_scale = static_cast<float>(_image_height) / static_cast<float>(_camera_height);
        return {_focal_x * x_scale, _focal_y * y_scale, _center_x * x_scale, _center_y * y_scale};
    }

    Tensor Camera::load_and_get_image(int resize_factor, int max_width) {
        const ImageLoadParams params{
            .path = _image_path,
            .resize_factor = resize_factor,
            .max_width = max_width,
            .stream = _stream};

        auto image = load_image_cached(params);

        const auto shape = image.shape();
        _image_width = shape[2];
        _image_height = shape[1];

        if (image.device() != Device::CUDA) {
            image = image.to(Device::CUDA, _stream);
            if (_stream) {
                cudaStreamSynchronize(_stream);
            }
        }

        return image;
    }

    void Camera::load_image_size(int resize_factor, int max_width) {
        int w, h;
        if (_undistort_prepared) {
            w = _camera_width;
            h = _camera_height;
        } else {
            auto result = get_image_info(_image_path);
            w = std::get<0>(result);
            h = std::get<1>(result);
        }

        LOG_DEBUG("load_image_size(): Base dimensions: {}x{}, resize_factor={}, max_width={}",
                  w, h, resize_factor, max_width);

        if (resize_factor > 0) {
            if (w % resize_factor || h % resize_factor) {
                LOG_WARN("width or height are not divisible by resize_factor w {} h {} resize_factor {}", w, h, resize_factor);
            }
            _image_width = w / resize_factor;
            _image_height = h / resize_factor;
        } else {
            _image_width = w;
            _image_height = h;
        }

        LOG_DEBUG("load_image_size(): After resize_factor: {}x{}", _image_width, _image_height);

        if (max_width > 0 && (_image_width > max_width || _image_height > max_width)) {
            int old_width = _image_width;
            int old_height = _image_height;

            if (_image_width > _image_height) {
                _image_width = max_width;
                _image_height = (old_height * max_width) / old_width; // Fixed: Use old_width
                LOG_DEBUG("load_image_size(): Resized {}x{} → {}x{} (limited by max_width={})",
                          old_width, old_height, _image_width, _image_height, max_width);
            } else {
                _image_height = max_width;
                _image_width = (old_width * max_width) / old_height; // Fixed: Use old_height
                LOG_DEBUG("load_image_size(): Resized {}x{} → {}x{} (limited by max_width={})",
                          old_width, old_height, _image_width, _image_height, max_width);
            }
        }

        LOG_DEBUG("load_image_size(): Final dimensions: {}x{}", _image_width, _image_height);
    }

    size_t Camera::get_num_bytes_from_file(int resize_factor, int max_width) const {
        auto result = get_image_info(_image_path);

        int w = std::get<0>(result);
        int h = std::get<1>(result);
        int c = std::get<2>(result);

        if (resize_factor > 0) {
            w = w / resize_factor;
            h = h / resize_factor;
        }

        if (max_width > 0 && (w > max_width || h > max_width)) {
            if (w > h) {
                h = (h * max_width) / w;
                w = max_width;
            } else {
                w = (w * max_width) / h;
                h = max_width;
            }
        }

        size_t num_bytes = w * h * c * sizeof(float);
        return num_bytes;
    }

    size_t Camera::get_num_bytes_from_file() const {
        auto [w, h, c] = get_image_info(_image_path);
        size_t num_bytes = w * h * c * sizeof(float);
        return num_bytes;
    }

    Tensor Camera::load_and_get_mask(const int resize_factor, const int max_width,
                                     const bool invert_mask, const float mask_threshold) {
        if (_mask_loaded && _cached_mask.is_valid()) {
            return _cached_mask;
        }

        if (_mask_path.empty() || !std::filesystem::exists(_mask_path)) {
            return Tensor();
        }

        const ImageLoadParams params{
            .path = _mask_path,
            .resize_factor = resize_factor,
            .max_width = max_width,
            .stream = _stream};

        auto mask = load_image_cached(params);

        if (mask.device() != Device::CUDA) {
            mask = mask.to(Device::CUDA, _stream);
            if (_stream) {
                cudaStreamSynchronize(_stream);
            }
        }

        // Convert RGB [C,H,W] to grayscale [H,W]
        if (mask.ndim() == 3 && mask.shape()[0] >= 3) {
            const auto r = mask.slice(0, 0, 1).squeeze(0);
            const auto g = mask.slice(0, 1, 2).squeeze(0);
            const auto b = mask.slice(0, 2, 3).squeeze(0);
            mask = (r + g + b) / 3.0f;
        } else if (mask.ndim() == 3 && mask.shape()[0] == 1) {
            mask = mask.squeeze(0);
        }

        if (invert_mask) {
            mask = Tensor::full(mask.shape(), 1.0f, mask.device()) - mask;
        }

        // Threshold: values >= threshold become 1.0
        if (mask_threshold > 0.0f && mask_threshold < 1.0f) {
            const auto ones = Tensor::full(mask.shape(), 1.0f, mask.device());
            const auto threshold_mask = mask.ge(mask_threshold);
            mask = ones.where(threshold_mask, mask);
        }

        if (_undistort_prepared) {
            const auto scaled = scale_undistort_params(
                _undistort_params,
                static_cast<int>(mask.shape()[1]),
                static_cast<int>(mask.shape()[0]));
            mask = undistort_mask(mask, scaled, _stream);
        }

        _cached_mask = mask;
        _mask_loaded = true;

        LOG_DEBUG("Loaded mask for {}: [{},{}]", _image_name, mask.shape()[0], mask.shape()[1]);

        return _cached_mask;
    }

    void Camera::precompute_undistortion(float blank_pixels) {
        if (_undistort_precomputed)
            return;
        if (!has_distortion())
            return;

        _undistort_params = compute_undistort_params(
            _focal_x, _focal_y, _center_x, _center_y,
            _camera_width, _camera_height,
            _radial_distortion, _tangential_distortion,
            _camera_model_type, blank_pixels);

        _undistort_precomputed = true;
    }

    void Camera::prepare_undistortion(float blank_pixels) {
        if (_undistort_prepared)
            return;

        precompute_undistortion(blank_pixels);
        if (!_undistort_precomputed)
            return;

        _focal_x = _undistort_params.dst_fx;
        _focal_y = _undistort_params.dst_fy;
        _center_x = _undistort_params.dst_cx;
        _center_y = _undistort_params.dst_cy;
        _camera_width = _undistort_params.dst_width;
        _camera_height = _undistort_params.dst_height;
        _FoVx = focal2fov(_focal_x, _camera_width);
        _FoVy = focal2fov(_focal_y, _camera_height);
        _undistort_prepared = true;
    }

    bool Camera::has_distortion() const noexcept {
        if (_radial_distortion.is_valid() && _radial_distortion.numel() > 0)
            return true;
        if (_tangential_distortion.is_valid() && _tangential_distortion.numel() > 0)
            return true;
        if (_camera_model_type != CameraModelType::PINHOLE)
            return true;
        return false;
    }

} // namespace lfs::core