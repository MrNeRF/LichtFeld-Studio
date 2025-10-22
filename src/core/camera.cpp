/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "loader/cache_image_loader.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

using torch::indexing::None;
using torch::indexing::Slice;

namespace gs {
    static torch::Tensor world_to_view(const torch::Tensor& R, const torch::Tensor& t) {
        torch::Tensor w2c = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(R.device()));
        w2c.index_put_({Slice(0, 3), Slice(0, 3)}, R);

        w2c.index_put_({Slice(0, 3), 3}, t);

        return w2c.to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).unsqueeze(0).contiguous();
    }

    Camera::Camera(const torch::Tensor& R,
                   const torch::Tensor& T,
                   float focal_x, float focal_y,
                   float center_x, float center_y,
                   const torch::Tensor radial_distortion,
                   const torch::Tensor tangential_distortion,
                   gsplat::CameraModelType camera_model_type,
                   const std::string& image_name,
                   const std::filesystem::path& image_path,
                   int camera_width, int camera_height,
                   int uid)
        : _uid(uid),
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
          _camera_width(camera_width),
          _camera_height(camera_height),
          _image_width(camera_width),
          _image_height(camera_height),
          _world_view_transform{world_to_view(R, T)} {

        auto c2w = torch::inverse(_world_view_transform.squeeze());
        _cam_position = c2w.index({Slice(None, 3), 3}).contiguous().squeeze();
        _FoVx = focal2fov(_focal_x, _camera_width);
        _FoVy = focal2fov(_focal_y, _camera_height);
    }
    Camera::Camera(const Camera& other, const torch::Tensor& transform)
        : _uid(other._uid),
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
          _camera_width(other._camera_width),
          _camera_height(other._camera_height),
          _image_width(other._image_width),
          _image_height(other._image_height),
          _cam_position(other._cam_position),
          _FoVx(other._FoVx),
          _FoVy(other._FoVy) {
        _world_view_transform = transform;
    }
    torch::Tensor Camera::K() const {
        const auto K = torch::zeros({1, 3, 3}, _world_view_transform.options());
        auto [fx, fy, cx, cy] = get_intrinsics();
        K[0][0][0] = fx;
        K[0][1][1] = fy;
        K[0][0][2] = cx;
        K[0][1][2] = cy;
        K[0][2][2] = 1.0f;
        return K;
    }

    std::tuple<float, float, float, float> Camera::get_intrinsics() const {
        float x_scale_factor = float(_image_width) / float(_camera_width);
        float y_scale_factor = float(_image_height) / float(_camera_height);
        float fx = _focal_x * x_scale_factor;
        float fy = _focal_y * y_scale_factor;
        float cx = _center_x * x_scale_factor;
        float cy = _center_y * y_scale_factor;
        return std::make_tuple(fx, fy, cx, cy);
    }

    torch::Tensor Camera::load_and_get_image(int resize_factor, int max_width) {
        // Use pinned memory for faster GPU transfer
        auto pinned_options = torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true);

        unsigned char* data;
        int w, h, c;
        auto& loader = gs::loader::CacheLoader::getInstance();
        // Load image synchronously
        gs::loader::LoadParams params{.resize_factor = resize_factor, .max_width = max_width};

        auto result = loader.load_cached_image(_image_path, params);

        data = std::get<0>(result);
        w = std::get<1>(result);
        h = std::get<2>(result);
        c = std::get<3>(result);

        _image_width = w;
        _image_height = h;

        // Create tensor from pinned memory and transfer asynchronously
        torch::Tensor image = torch::from_blob(
            data,
            {h, w, c},
            {w * c, c, 1},
            pinned_options);

        // Use the CUDA stream for async transfer
        at::cuda::CUDAStreamGuard guard(_stream);

        image = image.to(torch::kCUDA, /*non_blocking=*/true)
                    .permute({2, 0, 1})
                    .to(torch::kFloat32) /
                255.0f;

        // Free the original data
        free_image(data);

        // Ensure the transfer is complete before returning
        _stream.synchronize();

        return image;
    }

    void Camera::load_image_size(int resize_factor, int max_width) {
        auto result = get_image_info(_image_path);

        int w = std::get<0>(result);
        int h = std::get<1>(result);

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

        if (max_width > 0 && (_image_width > max_width || _image_height > max_width)) {
            if (_image_width > _image_height) {
                _image_width = max_width;
                _image_height = (_image_height * max_width) / _image_width;
            } else {
                _image_height = max_width;
                _image_width = (_image_width * max_width) / _image_height;
            }
        }
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
} // namespace gs